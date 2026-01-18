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
class DataServicerProxy(ray_client_pb2_grpc.RayletDataStreamerServicer):

    def __init__(self, proxy_manager: ProxyManager):
        self.num_clients = 0
        self.clients_last_seen: Dict[str, float] = {}
        self.reconnect_grace_periods: Dict[str, float] = {}
        self.clients_lock = Lock()
        self.proxy_manager = proxy_manager
        self.stopped = Event()

    def modify_connection_info_resp(self, init_resp: ray_client_pb2.DataResponse) -> ray_client_pb2.DataResponse:
        """
        Modify the `num_clients` returned the ConnectionInfoResponse because
        individual SpecificServers only have **one** client.
        """
        init_type = init_resp.WhichOneof('type')
        if init_type != 'connection_info':
            return init_resp
        modified_resp = ray_client_pb2.DataResponse()
        modified_resp.CopyFrom(init_resp)
        with self.clients_lock:
            modified_resp.connection_info.num_clients = self.num_clients
        return modified_resp

    def Datapath(self, request_iterator, context):
        request_iterator = RequestIteratorProxy(request_iterator)
        cleanup_requested = False
        start_time = time.time()
        client_id = _get_client_id_from_context(context)
        if client_id == '':
            return
        reconnecting = _get_reconnecting_from_context(context)
        if reconnecting:
            with self.clients_lock:
                if client_id not in self.clients_last_seen:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details('Attempted to reconnect a session that has already been cleaned up')
                    return
                self.clients_last_seen[client_id] = start_time
            server = self.proxy_manager._get_server_for_client(client_id)
            channel = self.proxy_manager.get_channel(client_id)
            new_iter = request_iterator
        else:
            server = self.proxy_manager.create_specific_server(client_id)
            with self.clients_lock:
                self.clients_last_seen[client_id] = start_time
                self.num_clients += 1
        try:
            if not reconnecting:
                logger.info(f'New data connection from client {client_id}: ')
                init_req = next(request_iterator)
                with self.clients_lock:
                    self.reconnect_grace_periods[client_id] = init_req.init.reconnect_grace_period
                try:
                    modified_init_req, job_config = prepare_runtime_init_req(init_req)
                    if not self.proxy_manager.start_specific_server(client_id, job_config):
                        logger.error(f'Server startup failed for client: {client_id}, using JobConfig: {job_config}!')
                        raise RuntimeError(f'Starting Ray client server failed. See ray_client_server_{server.port}.err for detailed logs.')
                    channel = self.proxy_manager.get_channel(client_id)
                    if channel is None:
                        logger.error(f'Channel not found for {client_id}')
                        raise RuntimeError(f'Proxy failed to Connect to backend! Check `ray_client_server.err` and `ray_client_server_{server.port}.err` on the head node of the cluster for the relevant logs. By default these are located at /tmp/ray/session_latest/logs.')
                except Exception:
                    init_resp = ray_client_pb2.DataResponse(init=ray_client_pb2.InitResponse(ok=False, msg=traceback.format_exc()))
                    init_resp.req_id = init_req.req_id
                    yield init_resp
                    return None
                new_iter = chain([modified_init_req], request_iterator)
            stub = ray_client_pb2_grpc.RayletDataStreamerStub(channel)
            metadata = [('client_id', client_id), ('reconnecting', str(reconnecting))]
            resp_stream = stub.Datapath(new_iter, metadata=metadata)
            for resp in resp_stream:
                resp_type = resp.WhichOneof('type')
                if resp_type == 'connection_cleanup':
                    cleanup_requested = True
                yield self.modify_connection_info_resp(resp)
        except Exception as e:
            logger.exception('Proxying Datapath failed!')
            recoverable = _propagate_error_in_context(e, context)
            if not recoverable:
                cleanup_requested = True
        finally:
            cleanup_delay = self.reconnect_grace_periods.get(client_id)
            if not cleanup_requested and cleanup_delay is not None:
                self.stopped.wait(timeout=cleanup_delay)
            with self.clients_lock:
                if client_id not in self.clients_last_seen:
                    logger.info(f'{client_id} not found. Skipping clean up.')
                    return
                last_seen = self.clients_last_seen[client_id]
                logger.info(f'{client_id} last started stream at {last_seen}. Current stream started at {start_time}.')
                if last_seen > start_time:
                    logger.info('Client reconnected. Skipping cleanup.')
                    return
                logger.debug(f'Client detached: {client_id}')
                self.num_clients -= 1
                del self.clients_last_seen[client_id]
                if client_id in self.reconnect_grace_periods:
                    del self.reconnect_grace_periods[client_id]
                server.set_result(None)