import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _remove_redundant_initializers(models: List[ModelProto], name_sharing_dict: Dict[Tuple[str, int], str]):
    """
    TODO: short documentation.
    """
    to_pop = []
    for i in range(len(models)):
        for idx, initializer in enumerate(models[i].graph.initializer):
            if initializer.name != name_sharing_dict[initializer.name, i]:
                to_pop.append(idx)
        for idx in sorted(to_pop, reverse=True):
            models[i].graph.initializer.pop(idx)