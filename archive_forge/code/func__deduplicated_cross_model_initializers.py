import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _deduplicated_cross_model_initializers(models: List[ModelProto], suffix: str=None):
    """
    TODO: short documentation.
    """
    duplicates = _find_duplicate_initializers(models)
    name_sharing_dict = _create_name_sharing_dict(duplicates, suffix=suffix)
    _replace_input_names(models, name_sharing_dict)
    deduplicated_initializers = []
    deduplicated_name = set()
    for i in range(len(models)):
        for initializer in models[i].graph.initializer:
            name_id_pair = (initializer.name, i)
            if name_id_pair in name_sharing_dict and name_sharing_dict[name_id_pair] not in deduplicated_name:
                deduplicated_name.add(name_sharing_dict[name_id_pair])
                initializer.name = name_sharing_dict[name_id_pair]
                deduplicated_initializers.append(initializer)
    return deduplicated_initializers