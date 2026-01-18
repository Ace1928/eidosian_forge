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
def _optimize_graph(meta_graph_def, signature_def):
    """Optimize `meta_graph_def` using grappler.  Returns a `GraphDef`."""
    new_meta_graph_def = copy.deepcopy(meta_graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for tensor_info in list(signature_def.inputs.values()) + list(signature_def.outputs.values()):
        fetch_collection.node_list.value.append(tensor_info.name)
    new_meta_graph_def.collection_def['train_op'].CopyFrom(fetch_collection)
    new_meta_graph_def.ClearField('saver_def')
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.min_graph_nodes = -1
    return tf_optimizer.OptimizeGraph(config, new_meta_graph_def)