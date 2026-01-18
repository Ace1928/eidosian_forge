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
def _prune_removed_feed_nodes(signature_def, graph_def):
    """Identify the inputs in the signature no longer in graph_def, prune them.

  Args:
    signature_def: A `SignatureDef` instance.
    graph_def: A `GraphDef` instance.

  Returns:
    A new pruned `SignatureDef`.
  """
    node_names = set([n.name for n in graph_def.node])
    new_signature_def = meta_graph_pb2.SignatureDef()
    new_signature_def.CopyFrom(signature_def)
    for k, v in signature_def.inputs.items():
        tensor_name, _ = _parse_tensor_name(v.name)
        if tensor_name not in node_names:
            logging.warn("Signature input key '{}', tensor name '{}', has been pruned while freezing the graph.  Removing it from the compiled signatures.".format(k, tensor_name))
            del new_signature_def.inputs[k]
    return new_signature_def