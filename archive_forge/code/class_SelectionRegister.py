import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
@attr.frozen
class SelectionRegister(Register):
    """Register used to represent SELECT register for various LCU methods.

    `SelectionRegister` extends the `Register` class to store the iteration length
    corresponding to that register along with its size.

    LCU methods often make use of coherent for-loops via UnaryIteration, iterating over a range
    of values stored as a superposition over the `SELECT` register. Such (nested) coherent
    for-loops can be represented using a `Tuple[SelectionRegister, ...]` where the i'th entry
    stores the bitsize and iteration length of i'th nested for-loop.

    One useful feature when processing such nested for-loops is to flatten out a composite index,
    represented by a tuple of indices (i, j, ...), one for each selection register into a single
    integer that can be used to index a flat target register. An example of such a mapping
    function is described in Eq.45 of https://arxiv.org/abs/1805.03662. A general version of this
    mapping function can be implemented using `numpy.ravel_multi_index` and `numpy.unravel_index`.

    For example:
        1) We can flatten a 2D for-loop as follows
        >>> import numpy as np
        >>> N, M = 10, 20
        >>> flat_indices = set()
        >>> for x in range(N):
        ...     for y in range(M):
        ...         flat_idx = x * M + y
        ...         assert np.ravel_multi_index((x, y), (N, M)) == flat_idx
        ...         assert np.unravel_index(flat_idx, (N, M)) == (x, y)
        ...         flat_indices.add(flat_idx)
        >>> assert len(flat_indices) == N * M

        2) Similarly, we can flatten a 3D for-loop as follows
        >>> import numpy as np
        >>> N, M, L = 10, 20, 30
        >>> flat_indices = set()
        >>> for x in range(N):
        ...     for y in range(M):
        ...         for z in range(L):
        ...             flat_idx = x * M * L + y * L + z
        ...             assert np.ravel_multi_index((x, y, z), (N, M, L)) == flat_idx
        ...             assert np.unravel_index(flat_idx, (N, M, L)) == (x, y, z)
        ...             flat_indices.add(flat_idx)
        >>> assert len(flat_indices) == N * M * L
    """
    name: str
    bitsize: int
    iteration_length: int = attr.field()
    shape: Tuple[int, ...] = attr.field(converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=())
    side: Side = Side.THRU

    @iteration_length.default
    def _default_iteration_length(self):
        return 2 ** self.bitsize

    @iteration_length.validator
    def validate_iteration_length(self, attribute, value):
        if len(self.shape) != 0:
            raise ValueError(f'Selection register {self.name} should be flat. Found self.shape={self.shape!r}')
        if not 0 <= value <= 2 ** self.bitsize:
            raise ValueError(f'iteration length must be in range [0, 2^{self.bitsize}]')

    def __repr__(self) -> str:
        return f'cirq_ft.SelectionRegister(name="{self.name}", bitsize={self.bitsize}, shape={self.shape}, iteration_length={self.iteration_length})'