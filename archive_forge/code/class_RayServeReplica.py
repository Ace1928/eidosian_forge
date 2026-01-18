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
class RayServeReplica:
    """Handles requests with the provided callable."""

    def __init__(self, _callable: Callable, deployment_name: str, replica_tag: ReplicaTag, autoscaling_config: Any, version: DeploymentVersion, is_function: bool, controller_handle: ActorHandle, app_name: str) -> None:
        self.deployment_id = DeploymentID(deployment_name, app_name)
        self.replica_tag = replica_tag
        self.callable = _callable
        self.is_function = is_function
        self.version = version
        self.deployment_config: DeploymentConfig = version.deployment_config
        self.rwlock = aiorwlock.RWLock()
        self.delete_lock = asyncio.Lock()
        user_health_check = getattr(_callable, HEALTH_CHECK_METHOD, None)
        if not callable(user_health_check):

            def user_health_check():
                pass
        self.user_health_check = sync_to_async(user_health_check)
        self.request_counter = metrics.Counter('serve_deployment_request_counter', description='The number of queries that have been processed in this replica.', tag_keys=('route',))
        self.error_counter = metrics.Counter('serve_deployment_error_counter', description='The number of exceptions that have occurred in this replica.', tag_keys=('route',))
        self.restart_counter = metrics.Counter('serve_deployment_replica_starts', description='The number of times this replica has been restarted due to failure.')
        self.processing_latency_tracker = metrics.Histogram('serve_deployment_processing_latency_ms', description='The latency for queries to be processed.', boundaries=DEFAULT_LATENCY_BUCKET_MS, tag_keys=('route',))
        self.num_processing_items = metrics.Gauge('serve_replica_processing_queries', description='The current number of queries being processed.')
        self.num_pending_items = metrics.Gauge('serve_replica_pending_queries', description='The current number of pending queries.')
        self.restart_counter.inc()
        self.autoscaling_metrics_store = InMemoryMetricsStore()
        self.metrics_pusher = MetricsPusher()
        if autoscaling_config:
            process_remote_func = controller_handle.record_autoscaling_metrics.remote
            config = autoscaling_config
            self.metrics_pusher.register_task(self.collect_autoscaling_metrics, config.metrics_interval_s, process_remote_func)
            self.metrics_pusher.register_task(lambda: {self.replica_tag: self.get_num_pending_and_running_requests()}, min(RAY_SERVE_REPLICA_AUTOSCALING_METRIC_RECORD_PERIOD_S, config.metrics_interval_s), self._add_autoscaling_metrics_point)
        self.metrics_pusher.register_task(self._set_replica_requests_metrics, RAY_SERVE_GAUGE_METRIC_SET_PERIOD_S)
        self.metrics_pusher.start()

    def _add_autoscaling_metrics_point(self, data, send_timestamp: float):
        self.autoscaling_metrics_store.add_metrics_point(data, send_timestamp)

    def _set_replica_requests_metrics(self):
        self.num_processing_items.set(self.get_num_running_requests())
        self.num_pending_items.set(self.get_num_pending_requests())

    async def check_health(self):
        await self.user_health_check()

    def _get_handle_request_stats(self) -> Optional[Dict[str, int]]:
        replica_actor_name = self.deployment_id.to_replica_actor_class_name()
        actor_stats = ray.runtime_context.get_runtime_context()._get_actor_call_stats()
        method_stats = actor_stats.get(f'{replica_actor_name}.handle_request')
        streaming_method_stats = actor_stats.get(f'{replica_actor_name}.handle_request_streaming')
        method_stats_java = actor_stats.get(f'{replica_actor_name}.handle_request_from_java')
        return merge_dict(merge_dict(method_stats, streaming_method_stats), method_stats_java)

    def get_num_running_requests(self) -> int:
        stats = self._get_handle_request_stats() or {}
        return stats.get('running', 0)

    def get_num_pending_requests(self) -> int:
        stats = self._get_handle_request_stats() or {}
        return stats.get('pending', 0)

    def get_num_pending_and_running_requests(self) -> int:
        stats = self._get_handle_request_stats() or {}
        return stats.get('pending', 0) + stats.get('running', 0)

    def collect_autoscaling_metrics(self):
        look_back_period = self.deployment_config.autoscaling_config.look_back_period_s
        return (self.replica_tag, self.autoscaling_metrics_store.window_average(self.replica_tag, time.time() - look_back_period))

    def get_runner_method(self, request_metadata: RequestMetadata) -> Callable:
        method_name = request_metadata.call_method
        if not hasattr(self.callable, method_name):

            def callable_method_filter(attr):
                if attr.startswith('__'):
                    return False
                elif not callable(getattr(self.callable, attr)):
                    return False
                return True
            methods = list(filter(callable_method_filter, dir(self.callable)))
            raise RayServeException(f"Tried to call a method '{method_name}' that does not exist. Available methods: {methods}.")
        if self.is_function:
            return self.callable
        return getattr(self.callable, method_name)

    async def send_user_result_over_asgi(self, result: Any, scope: Scope, receive: Receive, send: Send):
        """Handle the result from user code and send it over the ASGI interface.

        If the result is already a Response type, it is sent directly. Otherwise, it
        is converted to a custom Response type that handles serialization for
        common Python objects.
        """
        if isinstance(result, starlette.responses.Response):
            await result(scope, receive, send)
        else:
            await Response(result).send(scope, receive, send)

    async def reconfigure(self, deployment_config: DeploymentConfig):
        old_user_config = self.deployment_config.user_config
        self.deployment_config = deployment_config
        self.version = DeploymentVersion.from_deployment_version(self.version, self.deployment_config)
        if deployment_config.logging_config:
            logging_config = LoggingConfig(**deployment_config.logging_config)
            configure_component_logger(component_type=ServeComponentType.REPLICA, component_name=self.deployment_id.name, component_id=self.replica_tag, logging_config=logging_config)
        if old_user_config != deployment_config.user_config:
            await self.update_user_config(deployment_config.user_config)

    async def update_user_config(self, user_config: Any):
        async with self.rwlock.writer:
            if user_config is not None:
                if self.is_function:
                    raise ValueError('deployment_def must be a class to use user_config')
                elif not hasattr(self.callable, RECONFIGURE_METHOD):
                    raise RayServeException('user_config specified but deployment ' + self.deployment_id + ' missing ' + RECONFIGURE_METHOD + ' method')
                reconfigure_method = sync_to_async(getattr(self.callable, RECONFIGURE_METHOD))
                await reconfigure_method(user_config)

    @asynccontextmanager
    async def wrap_user_method_call(self, request_metadata: RequestMetadata):
        """Context manager that should be used to wrap user method calls.

        This sets up the serve request context, grabs the reader lock to avoid mutating
        user_config during method calls, and records metrics based on the result of the
        method.
        """
        ray.serve.context._serve_request_context.set(ray.serve.context._RequestContext(request_metadata.route, request_metadata.request_id, self.deployment_id.app, request_metadata.multiplexed_model_id, request_metadata.grpc_context))
        logger.info(f'Started executing request {request_metadata.request_id}', extra={'log_to_stderr': False, 'serve_access_log': True})
        start_time = time.time()
        user_exception = None
        try:
            yield
        except Exception as e:
            user_exception = e
            logger.exception(f'Request failed due to {type(e).__name__}:')
            if ray.util.pdb._is_ray_debugger_enabled():
                ray.util.pdb._post_mortem()
        latency_ms = (time.time() - start_time) * 1000
        self.processing_latency_tracker.observe(latency_ms, tags={'route': request_metadata.route})
        if user_exception is None:
            status_str = 'OK'
        elif isinstance(user_exception, asyncio.CancelledError):
            status_str = 'CANCELLED'
        else:
            status_str = 'ERROR'
        logger.info(access_log_msg(method=request_metadata.call_method, status=status_str, latency_ms=latency_ms), extra={'serve_access_log': True})
        if user_exception is None:
            self.request_counter.inc(tags={'route': request_metadata.route})
        else:
            self.error_counter.inc(tags={'route': request_metadata.route})
            raise user_exception from None

    async def call_user_method_with_grpc_unary_stream(self, request_metadata: RequestMetadata, request: gRPCRequest) -> AsyncGenerator[Tuple[RayServegRPCContext, bytes], None]:
        """Call a user method that is expected to be a generator.

        Deserializes gRPC request into protobuf object and pass into replica's runner
        method. Returns a generator of serialized protobuf bytes from the replica.
        """
        async with self.wrap_user_method_call(request_metadata):
            user_method = self.get_runner_method(request_metadata)
            user_request = pickle.loads(request.grpc_user_request)
            if GRPC_CONTEXT_ARG_NAME in inspect.signature(user_method).parameters:
                result_generator = user_method(user_request, grpc_context=request_metadata.grpc_context)
            else:
                result_generator = user_method(user_request)
            if inspect.iscoroutine(result_generator):
                result_generator = await result_generator
            if inspect.isgenerator(result_generator):
                for result in result_generator:
                    yield (request_metadata.grpc_context, result.SerializeToString())
            elif inspect.isasyncgen(result_generator):
                async for result in result_generator:
                    yield (request_metadata.grpc_context, result.SerializeToString())
            else:
                raise TypeError(f"When using `stream=True`, the called method must be a generator function, but '{user_method.__name__}' is not.")

    async def call_user_method_grpc_unary(self, request_metadata: RequestMetadata, request: gRPCRequest) -> Tuple[RayServegRPCContext, bytes]:
        """Call a user method that is *not* expected to be a generator.

        Deserializes gRPC request into protobuf object and pass into replica's runner
        method. Returns a serialized protobuf bytes from the replica.
        """
        async with self.wrap_user_method_call(request_metadata):
            user_request = pickle.loads(request.grpc_user_request)
            runner_method = self.get_runner_method(request_metadata)
            if inspect.isgeneratorfunction(runner_method) or inspect.isasyncgenfunction(runner_method):
                raise TypeError(f"Method '{runner_method.__name__}' is a generator function. You must use `handle.options(stream=True)` to call generators on a deployment.")
            method_to_call = sync_to_async(runner_method)
            if GRPC_CONTEXT_ARG_NAME in inspect.signature(runner_method).parameters:
                result = await method_to_call(user_request, grpc_context=request_metadata.grpc_context)
            else:
                result = await method_to_call(user_request)
            return (request_metadata.grpc_context, result.SerializeToString())

    async def call_user_method(self, request_metadata: RequestMetadata, request_args: Tuple[Any], request_kwargs: Dict[str, Any]) -> Any:
        """Call a user method that is *not* expected to be a generator.

        Raises any exception raised by the user code so it can be propagated as a
        `RayTaskError`.
        """
        async with self.wrap_user_method_call(request_metadata):
            if request_metadata.is_http_request:
                assert len(request_args) == 3
                scope, receive, send = request_args
                if isinstance(self.callable, ASGIAppReplicaWrapper):
                    request_args = (scope, receive, send)
                else:
                    request_args = (Request(scope, receive, send),)
            runner_method = None
            try:
                runner_method = self.get_runner_method(request_metadata)
                if inspect.isgeneratorfunction(runner_method) or inspect.isasyncgenfunction(runner_method):
                    raise TypeError(f"Method '{runner_method.__name__}' is a generator function. You must use `handle.options(stream=True)` to call generators on a deployment.")
                method_to_call = sync_to_async(runner_method)
                if request_metadata.is_http_request and len(inspect.signature(runner_method).parameters) == 0:
                    request_args, request_kwargs = (tuple(), {})
                result = await method_to_call(*request_args, **request_kwargs)
                if inspect.isgenerator(result) or inspect.isasyncgen(result):
                    raise TypeError(f"Method '{runner_method.__name__}' returned a generator. You must use `handle.options(stream=True)` to call generators on a deployment.")
            except Exception as e:
                function_name = 'unknown'
                if runner_method is not None:
                    function_name = runner_method.__name__
                e = wrap_to_ray_error(function_name, e)
                if request_metadata.is_http_request:
                    result = starlette.responses.Response(f'Unexpected error, traceback: {e}.', status_code=500)
                    await self.send_user_result_over_asgi(result, scope, receive, send)
                raise e from None
            if request_metadata.is_http_request and (not isinstance(self.callable, ASGIAppReplicaWrapper)):
                await self.send_user_result_over_asgi(result, scope, receive, send)
            return result

    async def call_user_method_generator(self, request_metadata: RequestMetadata, request_args: Tuple[Any], request_kwargs: Dict[str, Any]) -> AsyncGenerator[Any, None]:
        """Call a user method that is expected to be a generator.

        Raises any exception raised by the user code so it can be propagated as a
        `RayTaskError`.
        """
        async with self.wrap_user_method_call(request_metadata):
            assert not request_metadata.is_http_request, 'HTTP requests should go through `call_user_method`.'
            user_method = self.get_runner_method(request_metadata)
            result_generator = user_method(*request_args, **request_kwargs)
            if inspect.iscoroutine(result_generator):
                result_generator = await result_generator
            if inspect.isgenerator(result_generator):
                for result in result_generator:
                    yield result
            elif inspect.isasyncgen(result_generator):
                async for result in result_generator:
                    yield result
            else:
                raise TypeError(f"When using `stream=True`, the called method must be a generator function, but '{user_method.__name__}' is not.")

    async def prepare_for_shutdown(self):
        """Perform graceful shutdown.

        Trigger a graceful shutdown protocol that will wait for all the queued
        tasks to be completed and return to the controller.
        """
        while True:
            await asyncio.sleep(self.deployment_config.graceful_shutdown_wait_loop_s)
            num_ongoing_requests = self.get_num_pending_and_running_requests()
            if num_ongoing_requests > 0:
                logger.info(f'Waiting for an additional {self.deployment_config.graceful_shutdown_wait_loop_s}s to shut down because there are {num_ongoing_requests} ongoing requests.')
            else:
                logger.info('Graceful shutdown complete; replica exiting.', extra={'log_to_stderr': False})
                break
        async with self.delete_lock:
            if self.metrics_pusher:
                self.metrics_pusher.shutdown()
            if not hasattr(self, 'callable'):
                return
            try:
                if hasattr(self.callable, '__del__'):
                    await sync_to_async(self.callable.__del__)()
                    setattr(self.callable, '__del__', lambda _: None)
                if hasattr(self.callable, '__serve_multiplex_wrapper'):
                    await getattr(self.callable, '__serve_multiplex_wrapper').shutdown()
            except Exception as e:
                logger.exception(f'Exception during graceful shutdown of replica: {e}')
            finally:
                if hasattr(self.callable, '__del__'):
                    del self.callable