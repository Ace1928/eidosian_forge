from typing import cast
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_mxnet
from thinc.types import Array1d, Array2d, IntsXd
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.fixture
def X(input_size: int) -> Array2d:
    ops: Ops = get_current_ops()
    return cast(Array2d, ops.alloc(shape=(1, input_size)))