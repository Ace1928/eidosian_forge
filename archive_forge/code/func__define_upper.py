from typing import List, Optional, cast
from thinc.api import Linear, Model, chain, list2array, use_ops, zero_init
from thinc.types import Floats2d
from ...compat import Literal
from ...errors import Errors
from ...tokens import Doc
from ...util import registry
from .._precomputable_affine import PrecomputableAffine
from ..tb_framework import TransitionModel
def _define_upper(nO, nI):
    return Linear(nO=nO, nI=nI, init_W=zero_init)