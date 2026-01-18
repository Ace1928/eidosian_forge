from __future__ import annotations
from typing import Mapping, Union, Tuple
from collections.abc import Iterable, Mapping as _Mapping
from itertools import chain, islice
import numpy as np
from numpy.typing import ArrayLike
from qiskit.circuit import Parameter, QuantumCircuit
from .shape import ShapedMixin, ShapeInput, shape_tuple
def _format_key(key: tuple[Parameter | str, ...]):
    return tuple(map(_param_name, key))