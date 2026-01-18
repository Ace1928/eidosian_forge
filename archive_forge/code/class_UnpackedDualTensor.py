import os
from collections import namedtuple
from typing import Any
import torch
from .grad_mode import _DecoratorContextManager
class UnpackedDualTensor(_UnpackedDualTensor):
    """Namedtuple returned by :func:`unpack_dual` containing the primal and tangent components of the dual tensor.

    See :func:`unpack_dual` for more details.

    """
    pass