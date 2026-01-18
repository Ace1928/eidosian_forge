import collections
import errno
import gc
import itertools
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
import numpy as np
from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
def _GetGraphDef(self, run_params, gdef_or_saved_model_dir):
    if isinstance(gdef_or_saved_model_dir, str):
        if run_params.is_v2:
            root = load.load(gdef_or_saved_model_dir)
            func = root.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            gdef = func.graph.as_graph_def()
            del func
            del root
            gc.collect()
            return gdef
        return saved_model_utils.get_meta_graph_def(gdef_or_saved_model_dir, tag_constants.SERVING).graph_def
    assert isinstance(gdef_or_saved_model_dir, graph_pb2.GraphDef), f'Incorrect `gdef_or_saved_model_dir` type, expected `graph_pb2.GraphDef`, but got: {type(gdef_or_saved_model_dir)}.'
    return gdef_or_saved_model_dir