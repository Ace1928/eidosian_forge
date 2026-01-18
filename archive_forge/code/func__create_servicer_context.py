import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
@contextlib.contextmanager
def _create_servicer_context(rpc_event, state, request_deserializer):
    from grpc import _server
    context = _server._Context(rpc_event, state, request_deserializer)
    yield context
    context._finalize_state()