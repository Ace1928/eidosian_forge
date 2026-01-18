import asyncio
import inspect
import logging
import os
import pickle
import time
import traceback
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Tuple
import aiorwlock
import starlette.responses
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send
import ray
from ray import cloudpickle
from ray._private.async_compat import sync_to_async
from ray._private.utils import get_or_create_event_loop
from ray.actor import ActorClass, ActorHandle
from ray.remote_function import RemoteFunction
from ray.serve import metrics
from ray.serve._private.autoscaling_metrics import InMemoryMetricsStore
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import CONTROL_PLANE_CONCURRENCY_GROUP
from ray.serve._private.http_util import (
from ray.serve._private.logging_utils import (
from ray.serve._private.router import RequestMetadata
from ray.serve._private.utils import (
from ray.serve._private.version import DeploymentVersion
from ray.serve.deployment import Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.grpc_util import RayServegRPCContext
from ray.serve.schema import LoggingConfig
class RayServeWrappedReplica(object):

    async def __init__(self, deployment_name, replica_tag, serialized_deployment_def: bytes, serialized_init_args: bytes, serialized_init_kwargs: bytes, deployment_config_proto_bytes: bytes, version: DeploymentVersion, controller_name: str, app_name: str=None):
        self._replica_tag = replica_tag
        deployment_config = DeploymentConfig.from_proto_bytes(deployment_config_proto_bytes)
        if deployment_config.logging_config is None:
            logging_config = LoggingConfig()
        else:
            logging_config = LoggingConfig(**deployment_config.logging_config)
        self._configure_logger_and_profilers(replica_tag, logging_config)
        self._event_loop = get_or_create_event_loop()
        deployment_def = cloudpickle.loads(serialized_deployment_def)
        if isinstance(deployment_def, str):
            import_path = deployment_def
            module_name, attr_name = parse_import_path(import_path)
            deployment_def = getattr(import_module(module_name), attr_name)
            if isinstance(deployment_def, RemoteFunction):
                deployment_def = deployment_def._function
            elif isinstance(deployment_def, ActorClass):
                deployment_def = deployment_def.__ray_metadata__.modified_class
            elif isinstance(deployment_def, Deployment):
                logger.warning(f'''The import path "{import_path}" contains a decorated Serve deployment. The decorator's settings are ignored when deploying via import path.''')
                deployment_def = deployment_def.func_or_class
        init_args = cloudpickle.loads(serialized_init_args)
        init_kwargs = cloudpickle.loads(serialized_init_kwargs)
        if inspect.isfunction(deployment_def):
            is_function = True
        elif inspect.isclass(deployment_def):
            is_function = False
        else:
            assert False, f"deployment_def must be function, class, or corresponding import path. Instead, it's type was {type(deployment_def)}."
        ray.serve.context._set_internal_replica_context(app_name=app_name, deployment=deployment_name, replica_tag=replica_tag, servable_object=None, controller_name=controller_name)
        assert controller_name, 'Must provide a valid controller_name'
        controller_handle = ray.get_actor(controller_name, namespace=SERVE_NAMESPACE)
        self._initialized = False

        async def initialize_replica():
            logger.info('Started initializing replica.', extra={'log_to_stderr': False})
            if is_function:
                _callable = deployment_def
            else:
                _callable = deployment_def.__new__(deployment_def)
                await sync_to_async(_callable.__init__)(*init_args, **init_kwargs)
                if isinstance(_callable, ASGIAppReplicaWrapper):
                    await _callable._run_asgi_lifespan_startup()
            ray.serve.context._set_internal_replica_context(app_name=app_name, deployment=deployment_name, replica_tag=replica_tag, servable_object=_callable, controller_name=controller_name)
            self.replica = RayServeReplica(_callable, deployment_name, replica_tag, deployment_config.autoscaling_config, version, is_function, controller_handle, app_name)
            self._initialized = True
            logger.info('Finished initializing replica.', extra={'log_to_stderr': False})
        self.replica = None
        self._initialize_replica = initialize_replica
        self._replica_init_lock = asyncio.Lock()

    def _configure_logger_and_profilers(self, replica_tag: ReplicaTag, logging_config: LoggingConfig):
        replica_name = ReplicaName.from_replica_tag(replica_tag)
        if replica_name.app_name:
            component_name = f'{replica_name.app_name}_{replica_name.deployment_name}'
        else:
            component_name = f'{replica_name.deployment_name}'
        component_id = replica_name.replica_suffix
        configure_component_logger(component_type=ServeComponentType.REPLICA, component_name=component_name, component_id=component_id, logging_config=logging_config)
        configure_component_memory_profiler(component_type=ServeComponentType.REPLICA, component_name=component_name, component_id=component_id)
        self.cpu_profiler, self.cpu_profiler_log = configure_component_cpu_profiler(component_type=ServeComponentType.REPLICA, component_name=component_name, component_id=component_id)

    @ray.method(concurrency_group=CONTROL_PLANE_CONCURRENCY_GROUP)
    def get_num_ongoing_requests(self) -> int:
        """Fetch the number of ongoing requests at this replica (queue length).

            This runs on a separate thread (using a Ray concurrency group) so it will
            not be blocked by user code.
            """
        return self.replica.get_num_pending_and_running_requests()

    async def handle_request(self, pickled_request_metadata: bytes, *request_args, **request_kwargs) -> Tuple[bytes, Any]:
        request_metadata = pickle.loads(pickled_request_metadata)
        if request_metadata.is_grpc_request:
            assert len(request_args) == 1 and isinstance(request_args[0], gRPCRequest)
            result = await self.replica.call_user_method_grpc_unary(request_metadata=request_metadata, request=request_args[0])
        else:
            result = await self.replica.call_user_method(request_metadata, request_args, request_kwargs)
        return result

    async def _handle_http_request_generator(self, request_metadata: RequestMetadata, request: StreamingHTTPRequest) -> AsyncGenerator[Message, None]:
        """Handle an HTTP request and stream ASGI messages to the caller.

            This is a generator that yields ASGI-compliant messages sent by user code
            via an ASGI send interface.
            """
        receiver_task = None
        call_user_method_task = None
        wait_for_message_task = None
        try:
            receiver = ASGIReceiveProxy(request_metadata.request_id, request.http_proxy_handle)
            receiver_task = self._event_loop.create_task(receiver.fetch_until_disconnect())
            scope = pickle.loads(request.pickled_asgi_scope)
            asgi_queue_send = ASGIMessageQueue()
            request_args = (scope, receiver, asgi_queue_send)
            request_kwargs = {}
            call_user_method_task = self._event_loop.create_task(self.replica.call_user_method(request_metadata, request_args, request_kwargs))
            while True:
                wait_for_message_task = self._event_loop.create_task(asgi_queue_send.wait_for_message())
                done, _ = await asyncio.wait([call_user_method_task, wait_for_message_task], return_when=asyncio.FIRST_COMPLETED)
                yield pickle.dumps(asgi_queue_send.get_messages_nowait())
                if call_user_method_task in done:
                    break
            e = call_user_method_task.exception()
            if e is not None:
                raise e from None
        finally:
            if receiver_task is not None:
                receiver_task.cancel()
            if call_user_method_task is not None and (not call_user_method_task.done()):
                call_user_method_task.cancel()
            if wait_for_message_task is not None and (not wait_for_message_task.done()):
                wait_for_message_task.cancel()

    async def handle_request_streaming(self, pickled_request_metadata: bytes, *request_args, **request_kwargs) -> AsyncGenerator[Any, None]:
        """Generator that is the entrypoint for all `stream=True` handle calls."""
        request_metadata = pickle.loads(pickled_request_metadata)
        if request_metadata.is_grpc_request:
            assert len(request_args) == 1 and isinstance(request_args[0], gRPCRequest)
            generator = self.replica.call_user_method_with_grpc_unary_stream(request_metadata, request_args[0])
        elif request_metadata.is_http_request:
            assert len(request_args) == 1 and isinstance(request_args[0], StreamingHTTPRequest)
            generator = self._handle_http_request_generator(request_metadata, request_args[0])
        else:
            generator = self.replica.call_user_method_generator(request_metadata, request_args, request_kwargs)
        async for result in generator:
            yield result

    async def handle_request_from_java(self, proto_request_metadata: bytes, *request_args, **request_kwargs) -> Any:
        from ray.serve.generated.serve_pb2 import RequestMetadata as RequestMetadataProto
        proto = RequestMetadataProto.FromString(proto_request_metadata)
        request_metadata: RequestMetadata = RequestMetadata(proto.request_id, proto.endpoint, call_method=proto.call_method, multiplexed_model_id=proto.multiplexed_model_id, route=proto.route)
        request_args = request_args[0]
        return await self.replica.call_user_method(request_metadata, request_args, request_kwargs)

    async def is_allocated(self) -> str:
        """poke the replica to check whether it's alive.

            When calling this method on an ActorHandle, it will complete as
            soon as the actor has started running. We use this mechanism to
            detect when a replica has been allocated a worker slot.
            At this time, the replica can transition from PENDING_ALLOCATION
            to PENDING_INITIALIZATION startup state.

            Returns:
                The PID, actor ID, node ID, node IP, and log filepath id of the replica.
            """
        return (os.getpid(), ray.get_runtime_context().get_actor_id(), ray.get_runtime_context().get_worker_id(), ray.get_runtime_context().get_node_id(), ray.util.get_node_ip_address(), get_component_logger_file_path())

    async def initialize_and_get_metadata(self, deployment_config: DeploymentConfig=None, _after: Optional[Any]=None) -> Tuple[DeploymentConfig, DeploymentVersion]:
        try:
            async with self._replica_init_lock:
                if not self._initialized:
                    await self._initialize_replica()
                if deployment_config:
                    await self.replica.update_user_config(deployment_config.user_config)
            await self.check_health()
            return await self._get_metadata()
        except Exception:
            raise RuntimeError(traceback.format_exc()) from None

    async def reconfigure(self, deployment_config: DeploymentConfig) -> Tuple[DeploymentConfig, DeploymentVersion]:
        try:
            await self.replica.reconfigure(deployment_config)
            return await self._get_metadata()
        except Exception:
            raise RuntimeError(traceback.format_exc()) from None

    async def _get_metadata(self) -> Tuple[DeploymentConfig, DeploymentVersion]:
        return (self.replica.version.deployment_config, self.replica.version)

    def _save_cpu_profile_data(self) -> str:
        """Saves CPU profiling data, if CPU profiling is enabled.

            Logs a warning if CPU profiling is disabled.
            """
        if self.cpu_profiler is not None:
            import marshal
            self.cpu_profiler.snapshot_stats()
            with open(self.cpu_profiler_log, 'wb') as f:
                marshal.dump(self.cpu_profiler.stats, f)
            logger.info(f'Saved CPU profile data to file "{self.cpu_profiler_log}"')
            return self.cpu_profiler_log
        else:
            logger.error('Attempted to save CPU profile data, but failed because no CPU profiler was running! Enable CPU profiling by enabling the RAY_SERVE_ENABLE_CPU_PROFILING env var.')

    async def prepare_for_shutdown(self):
        if self.replica is not None:
            return await self.replica.prepare_for_shutdown()

    @ray.method(concurrency_group=CONTROL_PLANE_CONCURRENCY_GROUP)
    async def check_health(self):
        await self.replica.check_health()