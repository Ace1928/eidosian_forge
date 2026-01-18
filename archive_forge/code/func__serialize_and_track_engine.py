import collections
from functools import partial  # pylint: disable=g-importing-member
import os
import platform
import sys
import tempfile
import numpy as np
import six as _six
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _serialize_and_track_engine(node):
    """Serialize TRT engines in the cache and track them."""
    canonical_engine_name = _get_canonical_engine_name(node.name)
    if canonical_engine_name in resource_map:
        return
    filename = os.path.join(engine_asset_dir, 'trt-serialized-engine.' + canonical_engine_name)
    try:
        gen_trt_ops.serialize_trt_resource(resource_name=canonical_engine_name, filename=filename, delete_resource=True, save_gpu_specific_engines=save_gpu_specific_engines)
    except errors.NotFoundError:
        logging.info('Could not find %s in TF-TRT cache. This can happen if build() is not called, which means TensorRT engines will be built and cached at runtime.', canonical_engine_name)
        return
    resource_map[canonical_engine_name] = _TRTEngineResource(canonical_engine_name, filename, self._conversion_params.maximum_cached_engines)