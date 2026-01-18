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
class RayletServicerProxy(ray_client_pb2_grpc.RayletDriverServicer):

    def __init__(self, ray_connect_handler: Callable, proxy_manager: ProxyManager):
        self.proxy_manager = proxy_manager
        self.ray_connect_handler = ray_connect_handler

    def _call_inner_function(self, request, context, method: str) -> Optional[ray_client_pb2_grpc.RayletDriverStub]:
        client_id = _get_client_id_from_context(context)
        chan = self.proxy_manager.get_channel(client_id)
        if not chan:
            logger.error(f'Channel for Client: {client_id} not found!')
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return None
        stub = ray_client_pb2_grpc.RayletDriverStub(chan)
        try:
            metadata = [('client_id', client_id)]
            if context:
                metadata = context.invocation_metadata()
            return getattr(stub, method)(request, metadata=metadata)
        except Exception as e:
            logger.exception(f'Proxying call to {method} failed!')
            _propagate_error_in_context(e, context)

    def _has_channel_for_request(self, context):
        client_id = _get_client_id_from_context(context)
        return self.proxy_manager.has_channel(client_id)

    def Init(self, request, context=None) -> ray_client_pb2.InitResponse:
        return self._call_inner_function(request, context, 'Init')

    def KVPut(self, request, context=None) -> ray_client_pb2.KVPutResponse:
        """Proxies internal_kv.put.

        This is used by the working_dir code to upload to the GCS before
        ray.init is called. In that case (if we don't have a server yet)
        we directly make the internal KV call from the proxier.

        Otherwise, we proxy the call to the downstream server as usual.
        """
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVPut')
        with disable_client_hook():
            already_exists = ray.experimental.internal_kv._internal_kv_put(request.key, request.value, overwrite=request.overwrite)
        return ray_client_pb2.KVPutResponse(already_exists=already_exists)

    def KVGet(self, request, context=None) -> ray_client_pb2.KVGetResponse:
        """Proxies internal_kv.get.

        This is used by the working_dir code to upload to the GCS before
        ray.init is called. In that case (if we don't have a server yet)
        we directly make the internal KV call from the proxier.

        Otherwise, we proxy the call to the downstream server as usual.
        """
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVGet')
        with disable_client_hook():
            value = ray.experimental.internal_kv._internal_kv_get(request.key)
        return ray_client_pb2.KVGetResponse(value=value)

    def KVDel(self, request, context=None) -> ray_client_pb2.KVDelResponse:
        """Proxies internal_kv.delete.

        This is used by the working_dir code to upload to the GCS before
        ray.init is called. In that case (if we don't have a server yet)
        we directly make the internal KV call from the proxier.

        Otherwise, we proxy the call to the downstream server as usual.
        """
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVDel')
        with disable_client_hook():
            ray.experimental.internal_kv._internal_kv_del(request.key)
        return ray_client_pb2.KVDelResponse()

    def KVList(self, request, context=None) -> ray_client_pb2.KVListResponse:
        """Proxies internal_kv.list.

        This is used by the working_dir code to upload to the GCS before
        ray.init is called. In that case (if we don't have a server yet)
        we directly make the internal KV call from the proxier.

        Otherwise, we proxy the call to the downstream server as usual.
        """
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVList')
        with disable_client_hook():
            keys = ray.experimental.internal_kv._internal_kv_list(request.prefix)
        return ray_client_pb2.KVListResponse(keys=keys)

    def KVExists(self, request, context=None) -> ray_client_pb2.KVExistsResponse:
        """Proxies internal_kv.exists.

        This is used by the working_dir code to upload to the GCS before
        ray.init is called. In that case (if we don't have a server yet)
        we directly make the internal KV call from the proxier.

        Otherwise, we proxy the call to the downstream server as usual.
        """
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVExists')
        with disable_client_hook():
            exists = ray.experimental.internal_kv._internal_kv_exists(request.key)
        return ray_client_pb2.KVExistsResponse(exists=exists)

    def PinRuntimeEnvURI(self, request, context=None) -> ray_client_pb2.ClientPinRuntimeEnvURIResponse:
        """Proxies internal_kv.pin_runtime_env_uri.

        This is used by the working_dir code to upload to the GCS before
        ray.init is called. In that case (if we don't have a server yet)
        we directly make the internal KV call from the proxier.

        Otherwise, we proxy the call to the downstream server as usual.
        """
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'PinRuntimeEnvURI')
        with disable_client_hook():
            ray.experimental.internal_kv._pin_runtime_env_uri(request.uri, expiration_s=request.expiration_s)
        return ray_client_pb2.ClientPinRuntimeEnvURIResponse()

    def ListNamedActors(self, request, context=None) -> ray_client_pb2.ClientListNamedActorsResponse:
        return self._call_inner_function(request, context, 'ListNamedActors')

    def ClusterInfo(self, request, context=None) -> ray_client_pb2.ClusterInfoResponse:
        if request.type == ray_client_pb2.ClusterInfoType.PING:
            resp = ray_client_pb2.ClusterInfoResponse(json=json.dumps({}))
            return resp
        return self._call_inner_function(request, context, 'ClusterInfo')

    def Terminate(self, req, context=None):
        return self._call_inner_function(req, context, 'Terminate')

    def GetObject(self, request, context=None):
        try:
            yield from self._call_inner_function(request, context, 'GetObject')
        except Exception as e:
            logger.exception('Proxying call to GetObject failed!')
            _propagate_error_in_context(e, context)

    def PutObject(self, request: ray_client_pb2.PutRequest, context=None) -> ray_client_pb2.PutResponse:
        return self._call_inner_function(request, context, 'PutObject')

    def WaitObject(self, request, context=None) -> ray_client_pb2.WaitResponse:
        return self._call_inner_function(request, context, 'WaitObject')

    def Schedule(self, task, context=None) -> ray_client_pb2.ClientTaskTicket:
        return self._call_inner_function(task, context, 'Schedule')