import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
def AppendGraphDebugInfo(self, fn_name, fn_debug_info):
    debug_info_str = fn_debug_info.SerializeToString()
    super().AppendGraphDebugInfo(fn_name, debug_info_str)