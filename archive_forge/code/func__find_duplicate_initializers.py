import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _find_duplicate_initializers(models: List[ModelProto]) -> DefaultDict[Tuple[int, str, Tuple], Set[Tuple[str, int]]]:
    """
    Creates a map (unique data) --> set of (initializer name, model id)

    Initializers with a dimension 0, or dimension 1 with data type int32 or int64, are not included in the generated map.
    """
    duplicates = defaultdict(set)
    for i in range(len(models)):
        for initializer in models[i].graph.initializer:
            tensor_dims = tuple(getattr(initializer, 'dims'))
            if len(tensor_dims) > 1 or (len(tensor_dims) == 1 and initializer.data_type not in [6, 7]):
                tensor_data = numpy_helper.to_array(initializer)
                hashed = hashlib.sha512()
                hashed.update(tensor_data)
                tensor_digest = hashed.hexdigest()
                duplicates[initializer.data_type, tensor_digest, tensor_dims].add((initializer.name, i))
    return duplicates