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
def _check_names_match(data_names, data_shapes, name, throw):
    """Check that input names matches input data descriptors."""
    actual = [x[0] for x in data_shapes]
    if sorted(data_names) != sorted(actual):
        msg = "Data provided by %s_shapes don't match names specified by %s_names (%s vs. %s)" % (name, name, str(data_shapes), str(data_names))
        if throw:
            raise ValueError(msg)
        else:
            warnings.warn(msg)