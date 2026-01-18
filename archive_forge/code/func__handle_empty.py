import warnings
from typing import Callable, List, Optional, Sequence, Tuple, cast
from thinc.api import Model, Ops, registry
from thinc.initializers import glorot_uniform_init
from thinc.types import Floats1d, Floats2d, Ints1d, Ragged
from thinc.util import partial
from ..attrs import ORTH
from ..errors import Errors, Warnings
from ..tokens import Doc
from ..vectors import Mode, Vectors
from ..vocab import Vocab
def _handle_empty(ops: Ops, nO: int):
    return (Ragged(ops.alloc2f(0, nO), ops.alloc1i(0)), lambda d_ragged: [])