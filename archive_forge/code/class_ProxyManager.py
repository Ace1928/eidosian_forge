import atexit
import json
import logging
import socket
import sys
import time
import traceback
from concurrent import futures
from dataclasses import dataclass
from itertools import chain
import urllib
from threading import Event, Lock, RLock, Thread
from typing import Callable, Dict, List, Optional, Tuple
import grpc
import psutil
import ray
import ray.core.generated.agent_manager_pb2 as agent_manager_pb2
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
import ray.core.generated.runtime_env_agent_pb2 as runtime_env_agent_pb2
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.parameter import RayParams
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.services import ProcessInfo, start_ray_client_server
from ray._private.tls_utils import add_port_to_grpc_server
from ray._private.utils import detect_fate_sharing_support
from ray.cloudpickle.compat import pickle
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import _get_reconnecting_from_context
class ProxyManager:

    def __init__(self, address: Optional[str], runtime_env_agent_address: str, *, session_dir: Optional[str]=None, redis_password: Optional[str]=None, runtime_env_agent_port: int=0):
        self.servers: Dict[str, SpecificServer] = dict()
        self.server_lock = RLock()
        self._address = address
        self._redis_password = redis_password
        self._free_ports: List[int] = list(range(MIN_SPECIFIC_SERVER_PORT, MAX_SPECIFIC_SERVER_PORT))
        self._runtime_env_agent_address = runtime_env_agent_address
        self._check_thread = Thread(target=self._check_processes, daemon=True)
        self._check_thread.start()
        self.fate_share = bool(detect_fate_sharing_support())
        self._node: Optional[ray._private.node.Node] = None
        atexit.register(self._cleanup)

    def _get_unused_port(self) -> int:
        """
        Search for a port in _free_ports that is unused.
        """
        with self.server_lock:
            num_ports = len(self._free_ports)
            for _ in range(num_ports):
                port = self._free_ports.pop(0)
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.bind(('', port))
                except OSError:
                    self._free_ports.append(port)
                    continue
                finally:
                    s.close()
                return port
        raise RuntimeError('Unable to succeed in selecting a random port.')

    @property
    def address(self) -> str:
        """
        Returns the provided Ray bootstrap address, or creates a new cluster.
        """
        if self._address:
            return self._address
        connection_tuple = ray.init()
        self._address = connection_tuple['address']
        self._session_dir = connection_tuple['session_dir']
        return self._address

    @property
    def node(self) -> ray._private.node.Node:
        """Gets a 'ray.Node' object for this node (the head node).
        If it does not already exist, one is created using the bootstrap
        address.
        """
        if self._node:
            return self._node
        ray_params = RayParams(gcs_address=self.address)
        self._node = ray._private.node.Node(ray_params, head=False, shutdown_at_exit=False, spawn_reaper=False, connect_only=True)
        return self._node

    def create_specific_server(self, client_id: str) -> SpecificServer:
        """
        Create, but not start a SpecificServer for a given client. This
        method must be called once per client.
        """
        with self.server_lock:
            assert self.servers.get(client_id) is None, f'Server already created for Client: {client_id}'
            port = self._get_unused_port()
            server = SpecificServer(port=port, process_handle_future=futures.Future(), channel=ray._private.utils.init_grpc_channel(f'127.0.0.1:{port}', options=GRPC_OPTIONS))
            self.servers[client_id] = server
            return server

    def _create_runtime_env(self, serialized_runtime_env: str, runtime_env_config: str, specific_server: SpecificServer):
        """Increase the runtime_env reference by sending an RPC to the agent.

        Includes retry logic to handle the case when the agent is
        temporarily unreachable (e.g., hasn't been started up yet).
        """
        logger.info(f'Increasing runtime env reference for ray_client_server_{specific_server.port}.Serialized runtime env is {serialized_runtime_env}.')
        assert len(self._runtime_env_agent_address) > 0, 'runtime_env_agent_address not set'
        create_env_request = runtime_env_agent_pb2.GetOrCreateRuntimeEnvRequest(serialized_runtime_env=serialized_runtime_env, runtime_env_config=runtime_env_config, job_id=f'ray_client_server_{specific_server.port}'.encode('utf-8'), source_process='client_server')
        retries = 0
        max_retries = 5
        wait_time_s = 0.5
        last_exception = None
        while retries <= max_retries:
            try:
                url = urllib.parse.urljoin(self._runtime_env_agent_address, '/get_or_create_runtime_env')
                data = create_env_request.SerializeToString()
                req = urllib.request.Request(url, data=data, method='POST')
                req.add_header('Content-Type', 'application/octet-stream')
                response = urllib.request.urlopen(req, timeout=None)
                response_data = response.read()
                r = runtime_env_agent_pb2.GetOrCreateRuntimeEnvReply()
                r.ParseFromString(response_data)
                if r.status == agent_manager_pb2.AgentRpcStatus.AGENT_RPC_STATUS_OK:
                    return r.serialized_runtime_env_context
                elif r.status == agent_manager_pb2.AgentRpcStatus.AGENT_RPC_STATUS_FAILED:
                    raise RuntimeError(f'Failed to create runtime_env for Ray client server, it is caused by:\n{r.error_message}')
                else:
                    assert False, f'Unknown status: {r.status}.'
            except urllib.error.URLError as e:
                last_exception = e
                logger.warning(f'GetOrCreateRuntimeEnv request failed: {e}. Retrying after {wait_time_s}s. {max_retries - retries} retries remaining.')
            time.sleep(wait_time_s)
            retries += 1
            wait_time_s *= 2
        raise TimeoutError(f'GetOrCreateRuntimeEnv request failed after {max_retries} attempts. Last exception: {last_exception}')

    def start_specific_server(self, client_id: str, job_config: JobConfig) -> bool:
        """
        Start up a RayClient Server for an incoming client to
        communicate with. Returns whether creation was successful.
        """
        specific_server = self._get_server_for_client(client_id)
        assert specific_server, f'Server has not been created for: {client_id}'
        output, error = self.node.get_log_file_handles(f'ray_client_server_{specific_server.port}', unique=True)
        serialized_runtime_env = job_config._get_serialized_runtime_env()
        runtime_env_config = job_config._get_proto_runtime_env_config()
        if not serialized_runtime_env or serialized_runtime_env == '{}':
            serialized_runtime_env_context = RuntimeEnvContext().serialize()
        else:
            serialized_runtime_env_context = self._create_runtime_env(serialized_runtime_env=serialized_runtime_env, runtime_env_config=runtime_env_config, specific_server=specific_server)
        proc = start_ray_client_server(self.address, self.node.node_ip_address, specific_server.port, stdout_file=output, stderr_file=error, fate_share=self.fate_share, server_type='specific-server', serialized_runtime_env_context=serialized_runtime_env_context, redis_password=self._redis_password)
        pid = proc.process.pid
        if sys.platform != 'win32':
            psutil_proc = psutil.Process(pid)
        else:
            psutil_proc = None
        while psutil_proc is not None:
            if proc.process.poll() is not None:
                logger.error(f'SpecificServer startup failed for client: {client_id}')
                break
            cmd = psutil_proc.cmdline()
            if _match_running_client_server(cmd):
                break
            logger.debug('Waiting for Process to reach the actual client server.')
            time.sleep(0.5)
        specific_server.set_result(proc)
        logger.info(f'SpecificServer started on port: {specific_server.port} with PID: {pid} for client: {client_id}')
        return proc.process.poll() is None

    def _get_server_for_client(self, client_id: str) -> Optional[SpecificServer]:
        with self.server_lock:
            client = self.servers.get(client_id)
            if client is None:
                logger.error(f'Unable to find channel for client: {client_id}')
            return client

    def has_channel(self, client_id: str) -> bool:
        server = self._get_server_for_client(client_id)
        if server is None:
            return False
        return server.is_ready()

    def get_channel(self, client_id: str) -> Optional['grpc._channel.Channel']:
        """
        Find the gRPC Channel for the given client_id. This will block until
        the server process has started.
        """
        server = self._get_server_for_client(client_id)
        if server is None:
            return None
        server.wait_ready()
        try:
            grpc.channel_ready_future(server.channel).result(timeout=CHECK_CHANNEL_TIMEOUT_S)
            return server.channel
        except grpc.FutureTimeoutError:
            logger.exception(f'Timeout waiting for channel for {client_id}')
            return None

    def _check_processes(self):
        """
        Keeps the internal servers dictionary up-to-date with running servers.
        """
        while True:
            with self.server_lock:
                for client_id, specific_server in list(self.servers.items()):
                    if specific_server.poll() is not None:
                        logger.info(f'Specific server {client_id} is no longer running, freeing its port {specific_server.port}')
                        del self.servers[client_id]
                        self._free_ports.append(specific_server.port)
            time.sleep(CHECK_PROCESS_INTERVAL_S)

    def _cleanup(self) -> None:
        """
        Forcibly kill all spawned RayClient Servers. This ensures cleanup
        for platforms where fate sharing is not supported.
        """
        for server in self.servers.values():
            server.kill()