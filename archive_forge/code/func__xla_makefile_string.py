import collections
import copy
import os
import re
import shlex
from typing import List, Tuple
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import versions
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
def _xla_makefile_string(output_prefix):
    """Returns a Makefile string with variables for using XLA binary object files.

  Attempts to identify the right include header paths when run from either
  an installed TensorFlow pip package, or from bazel run.

  Args:
    output_prefix: A string containing the output prefix for the XLA AOT
      compiled header + object files.

  Returns:
    A string containing a filled out `_XLA_MAKEFILE_TEMPLATE`.
  """
    sysconfig = _sysconfig_module()
    output_dir, _ = os.path.split(output_prefix)
    if sysconfig:
        tensorflow_includes = _shlex_quote(sysconfig.get_include())
    else:
        if os.path.islink(__file__):
            this_file = __file__
            while os.path.islink(this_file):
                this_file = os.readlink(this_file)
            base = os.path.realpath(os.path.join(os.path.dirname(this_file), *[os.path.pardir] * 3))
        else:
            try:
                base = test.test_src_dir_path('')
            except KeyError:
                base = os.path.realpath(os.path.join(os.path.dirname(__file__), *[os.path.pardir] * 3))
        expected_header = os.path.join(base, 'tensorflow', 'compiler', 'tf2xla', 'xla_compiled_cpu_function.h')
        if not os.path.exists(expected_header):
            logging.error('Could not find includes path.  Missing file: {}'.format(expected_header))
        tensorflow_includes = base
    return _XLA_MAKEFILE_TEMPLATE.format(tensorflow_includes=tensorflow_includes, compiled_dir=_shlex_quote(output_dir), cxx_flags='-D_GLIBCXX_USE_CXX11_ABI={}'.format(versions.CXX11_ABI_FLAG))