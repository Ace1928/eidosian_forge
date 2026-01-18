import time
import logging
import warnings
import numpy as np
from .. import metric
from .. import ndarray
from ..context import cpu
from ..model import BatchEndParam
from ..initializer import Uniform
from ..io import DataDesc, DataIter, DataBatch
from ..base import _as_list
def _parse_data_desc(data_names, label_names, data_shapes, label_shapes):
    """parse data_attrs into DataDesc format and check that names match"""
    data_shapes = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in data_shapes]
    _check_names_match(data_names, data_shapes, 'data', True)
    if label_shapes is not None:
        label_shapes = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in label_shapes]
        _check_names_match(label_names, label_shapes, 'label', False)
    else:
        _check_names_match(label_names, [], 'label', False)
    return (data_shapes, label_shapes)