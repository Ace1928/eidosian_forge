import collections
import threading
import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import abandonment
from grpc.framework.foundation import logging_pool
from grpc.framework.foundation import stream
from grpc.framework.interfaces.face import face
def _flatten_method_pair_map(method_pair_map):
    method_pair_map = method_pair_map or {}
    flat_map = {}
    for method_pair in method_pair_map:
        method = _common.fully_qualified_method(method_pair[0], method_pair[1])
        flat_map[method] = method_pair_map[method_pair]
    return flat_map