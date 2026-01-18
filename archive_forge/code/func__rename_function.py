import math
import numbers
import os
import re
import sys
import time
import types
from absl import app
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _rename_function(f, arg_num, name):
    """Rename the given function's name appears in the stack trace."""
    func_code = f.__code__
    new_code = func_code.replace(co_argcount=arg_num, co_name=name)
    return types.FunctionType(new_code, f.__globals__, name, f.__defaults__, f.__closure__)