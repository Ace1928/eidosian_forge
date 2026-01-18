import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _replace_input_names(models: List[ModelProto], name_sharing_dict: Dict[Tuple[str, int], str]):
    """
    Replaces the names of node inputs from the models by the names in the name_sharing_dict.
    """
    for i in range(len(models)):
        for node in models[i].graph.node:
            for j in range(len(node.input)):
                if (node.input[j], i) in name_sharing_dict:
                    node.input[j] = name_sharing_dict[node.input[j], i]