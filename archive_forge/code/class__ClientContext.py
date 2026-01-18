import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
import ray._private.ray_constants as ray_constants
from ray._private.client_mode_hook import (
from ray._private.ray_logging import setup_logger
from ray.job_config import JobConfig
from ray.util.annotations import DeveloperAPI
class _ClientContext:

    def __init__(self):
        from ray.util.client.api import _ClientAPI
        self.api = _ClientAPI()
        self.client_worker = None
        self._server = None
        self._connected_with_init = False
        self._inside_client_test = False

    def connect(self, conn_str: str, job_config: JobConfig=None, secure: bool=False, metadata: List[Tuple[str, str]]=None, connection_retries: int=3, namespace: str=None, *, ignore_version: bool=False, _credentials: Optional['grpc.ChannelCredentials']=None, ray_init_kwargs: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """Connect the Ray Client to a server.

        Args:
            conn_str: Connection string, in the form "[host]:port"
            job_config: The job config of the server.
            secure: Whether to use a TLS secured gRPC channel
            metadata: gRPC metadata to send on connect
            connection_retries: number of connection attempts to make
            ignore_version: whether to ignore Python or Ray version mismatches.
                This should only be used for debugging purposes.

        Returns:
            Dictionary of connection info, e.g., {"num_clients": 1}.
        """
        from ray.util.client.worker import Worker
        if self.client_worker is not None:
            if self._connected_with_init:
                return
            raise Exception('ray.init() called, but ray client is already connected')
        if not self._inside_client_test:
            _explicitly_enable_client_mode()
        if namespace is not None:
            job_config = job_config or JobConfig()
            job_config.set_ray_namespace(namespace)
        logging_level = ray_constants.LOGGER_LEVEL
        logging_format = ray_constants.LOGGER_FORMAT
        if ray_init_kwargs is None:
            ray_init_kwargs = {}
        ray_init_kwargs['_skip_env_hook'] = True
        if ray_init_kwargs.get('logging_level') is not None:
            logging_level = ray_init_kwargs['logging_level']
        if ray_init_kwargs.get('logging_format') is not None:
            logging_format = ray_init_kwargs['logging_format']
        setup_logger(logging_level, logging_format)
        try:
            self.client_worker = Worker(conn_str, secure=secure, _credentials=_credentials, metadata=metadata, connection_retries=connection_retries)
            self.api.worker = self.client_worker
            self.client_worker._server_init(job_config, ray_init_kwargs)
            conn_info = self.client_worker.connection_info()
            self._check_versions(conn_info, ignore_version)
            self._register_serializers()
            return conn_info
        except Exception:
            self.disconnect()
            raise

    def _register_serializers(self):
        """Register the custom serializer addons at the client side.

        The server side should have already registered the serializers via
        regular worker's serialization_context mechanism.
        """
        import ray.util.serialization_addons
        from ray.util.serialization import StandaloneSerializationContext
        ctx = StandaloneSerializationContext()
        ray.util.serialization_addons.apply(ctx)

    def _check_versions(self, conn_info: Dict[str, Any], ignore_version: bool) -> None:
        local_major_minor = f'{sys.version_info[0]}.{sys.version_info[1]}'
        if not conn_info['python_version'].startswith(local_major_minor):
            version_str = f'{local_major_minor}.{sys.version_info[2]}'
            msg = 'Python minor versions differ between client and server:' + f' client is {version_str},' + f' server is {conn_info['python_version']}'
            if ignore_version or 'RAY_IGNORE_VERSION_MISMATCH' in os.environ:
                logger.warning(msg)
            else:
                raise RuntimeError(msg)
        if CURRENT_PROTOCOL_VERSION != conn_info['protocol_version']:
            msg = 'Client Ray installation incompatible with server:' + f' client is {CURRENT_PROTOCOL_VERSION},' + f' server is {conn_info['protocol_version']}'
            if ignore_version or 'RAY_IGNORE_VERSION_MISMATCH' in os.environ:
                logger.warning(msg)
            else:
                raise RuntimeError(msg)

    def disconnect(self):
        """Disconnect the Ray Client."""
        from ray.util.client.api import _ClientAPI
        if self.client_worker is not None:
            self.client_worker.close()
        self.api = _ClientAPI()
        self.client_worker = None

    def remote(self, *args, **kwargs):
        """remote is the hook stub passed on to replace `ray.remote`.

        This sets up remote functions or actors, as the decorator,
        but does not execute them.

        Args:
            args: opaque arguments
            kwargs: opaque keyword arguments
        """
        return self.api.remote(*args, **kwargs)

    def __getattr__(self, key: str):
        if self.is_connected():
            return getattr(self.api, key)
        elif key in ['is_initialized', '_internal_kv_initialized']:
            return lambda: False
        else:
            raise Exception('Ray Client is not connected. Please connect by calling `ray.init`.')

    def is_connected(self) -> bool:
        if self.client_worker is None:
            return False
        return self.client_worker.is_connected()

    def init(self, *args, **kwargs):
        if self._server is not None:
            raise Exception('Trying to start two instances of ray via client')
        import ray.util.client.server.server as ray_client_server
        server_handle, address_info = ray_client_server.init_and_serve('127.0.0.1:50051', *args, **kwargs)
        self._server = server_handle.grpc_server
        self.connect('127.0.0.1:50051')
        self._connected_with_init = True
        return address_info

    def shutdown(self, _exiting_interpreter=False):
        self.disconnect()
        import ray.util.client.server.server as ray_client_server
        if self._server is None:
            return
        ray_client_server.shutdown_with_server(self._server, _exiting_interpreter)
        self._server = None