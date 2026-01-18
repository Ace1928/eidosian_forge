import copy
import logging
import os
import re
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from typing import Set, Dict, Tuple, List
def _remap_keys(old_dict, rename_fn):
    """
    Rename keys of 'old_dict' according to 'rename_fn'.

    Args:
        old_dict: Dictionary (i.e. containing blob_name -> blob_name
            relationships.)
        rename_fn: Function string -> string for renaming.

    Returns:
        None. Modifies old_dict in-place.
    """
    new_dict = {rename_fn(key): value for key, value in old_dict.items()}
    old_dict.clear()
    old_dict.update(new_dict)