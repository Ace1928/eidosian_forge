import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
def LoadTracesFromDebugInfo(debug_info):
    return _tf_stack.LoadTracesFromDebugInfo(debug_info.SerializeToString())