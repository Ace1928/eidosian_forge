from __future__ import annotations
from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping
import numpy as np
from numpy.typing import NDArray
from qiskit.result import Counts
from .shape import ShapedMixin, ShapeInput, shape_tuple
Return a new reshaped bit array.

        The :attr:`~num_shots` axis is either included or excluded from the reshaping procedure
        depending on which picture the new shape is compatible with. For example, for a bit array
        with shape ``(20, 5)`` and ``64`` shots, a reshape to ``(100,)`` would leave the
        number of shots intact, whereas a reshape to ``(200, 32)`` would change the number of
        shots to ``32``.

        Args:
            *shape: The new desired shape.

        Returns:
            A new bit array.

        Raises:
            ValueError: If the size corresponding to your new shape is not equal to either
                :attr:`~size`, or the product of :attr:`~size` and :attr:`~num_shots`.
        