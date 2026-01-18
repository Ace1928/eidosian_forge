import copy
from typing import Any, cast
import srsly
from ..compat import mxnet as mx
from ..optimizers import Optimizer
from ..types import ArgsKwargs, FloatsXd
from ..util import (
from .shim import Shim
def _load_params(self, params):
    with make_tempfile('w+b') as temp:
        temp.write(params)
        self._model.load_parameters(temp.name, ctx=mx.current_context())