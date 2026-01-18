import dataclasses
import traceback
import typing
from typing import Any, Dict, List, Optional, Sequence, Union
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_stack
class InterpolateRuntimeError(object):
    """Context Manager that interpolates exceptions received by AtomicFunction."""
    DENY_LIST_PHRASES = ['<embedded']

    def __init__(self, top_level_func):
        self._func = top_level_func

    def interpolate(self, message, node_names, graph_debug_info):
        """Uses the GraphDebugInfo to generate an error message."""
        error_message = ['Graph execution error:', '']
        traces = tf_stack.LoadTracesFromDebugInfo(graph_debug_info)
        for node_name in node_names:
            error_message.append(f'Detected at node {node_name} defined at (most recent call last):')
            if node_name in traces:
                stack_trace = traces[node_name]
                for formatted_frame in traceback.format_list(stack_trace):
                    if not any((p in formatted_frame for p in self.DENY_LIST_PHRASES)):
                        error_message.append(formatted_frame)
            else:
                error_message.append('<stack traces unavailable>')
        error_message.append(message.strip())
        return '\n'.join(error_message)

    def __enter__(self):
        pass

    def __exit__(self, typ, exc, tb):
        if not exc or not isinstance(exc, errors.OpError):
            return False
        exc = typing.cast(errors.OpError, exc)
        message = compat.as_text(exc.message)
        parsed_message, func_tags, node_tags = error_interpolation.parse_message(message)
        deepest_func = None
        for func_tag in func_tags:
            if func_tag.name == compat.as_str(self._func.name):
                deepest_func = self._func
            elif deepest_func:
                next_func = None
                for child_func in deepest_func.children:
                    if func_tag.name == compat.as_str(child_func.name):
                        next_func = child_func
                        break
                if next_func is not None and isinstance(next_func, AtomicFunction):
                    deepest_func = next_func
        if deepest_func:
            exc._message = self.interpolate(parsed_message, [t.name for t in node_tags], deepest_func.graph_debug_info)
        return False