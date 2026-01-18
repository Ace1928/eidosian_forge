import base64
import json
import logging
import os
import tempfile
import threading
import time
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import grpc
import ray._private.tls_utils
import ray.cloudpickle as cloudpickle
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private.ray_constants import DEFAULT_CLIENT_RECONNECT_GRACE_PERIOD
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.cloudpickle.compat import pickle
from ray.exceptions import GetTimeoutError
from ray.job_config import JobConfig
from ray.util.client.client_pickler import dumps_from_client, loads_from_server
from ray.util.client.common import (
from ray.util.client.dataclient import DataClient
from ray.util.client.logsclient import LogstreamClient
from ray.util.debug import log_once
def _connect_channel(self, reconnecting=False) -> None:
    """
        Attempts to connect to the server specified by conn_str. If
        reconnecting after an RPC error, cleans up the old channel and
        continues to attempt to connect until the grace period is over.
        """
    if self.channel is not None:
        self.channel.unsubscribe(self._on_channel_state_change)
        self.channel.close()
    if self._secure:
        if self._credentials is not None:
            credentials = self._credentials
        elif os.environ.get('RAY_USE_TLS', '0').lower() in ('1', 'true'):
            server_cert_chain, private_key, ca_cert = ray._private.tls_utils.load_certs_from_env()
            credentials = grpc.ssl_channel_credentials(certificate_chain=server_cert_chain, private_key=private_key, root_certificates=ca_cert)
        else:
            credentials = grpc.ssl_channel_credentials()
        self.channel = grpc.secure_channel(self._conn_str, credentials, options=GRPC_OPTIONS)
    else:
        self.channel = grpc.insecure_channel(self._conn_str, options=GRPC_OPTIONS)
    self.channel.subscribe(self._on_channel_state_change)
    start_time = time.time()
    conn_attempts = 0
    timeout = INITIAL_TIMEOUT_SEC
    service_ready = False
    while conn_attempts < max(self._connection_retries, 1) or reconnecting:
        conn_attempts += 1
        if self._in_shutdown:
            break
        elapsed_time = time.time() - start_time
        if reconnecting and elapsed_time > self._reconnect_grace_period:
            self._in_shutdown = True
            raise ConnectionError(f'Failed to reconnect within the reconnection grace period ({self._reconnect_grace_period}s)')
        try:
            grpc.channel_ready_future(self.channel).result(timeout=timeout)
            self.server = ray_client_pb2_grpc.RayletDriverStub(self.channel)
            service_ready = bool(self.ping_server())
            if service_ready:
                break
            time.sleep(timeout)
        except grpc.FutureTimeoutError:
            logger.debug(f"Couldn't connect channel in {timeout} seconds, retrying")
        except grpc.RpcError as e:
            logger.debug(f'Ray client server unavailable, retrying in {timeout}s...')
            logger.debug(f'Received when checking init: {e.details()}')
            time.sleep(timeout)
        logger.debug(f'Waiting for Ray to become ready on the server, retry in {timeout}s...')
        if not reconnecting:
            timeout = backoff(timeout)
    if not service_ready:
        self._in_shutdown = True
        if log_once('ray_client_security_groups'):
            warnings.warn('Ray Client connection timed out. Ensure that the Ray Client port on the head node is reachable from your local machine. See https://docs.ray.io/en/latest/cluster/ray-client.html#step-2-check-ports for more information.')
        raise ConnectionError('ray client connection timeout')