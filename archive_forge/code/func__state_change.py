import collections
import json
import queue
import threading
import time
from concurrent import futures
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _state_change(new_state, node_name, output_slot, debug_op):
    state_change = debug_service_pb2.EventReply.DebugOpStateChange()
    state_change.state = new_state
    state_change.node_name = node_name
    state_change.output_slot = output_slot
    state_change.debug_op = debug_op
    return state_change