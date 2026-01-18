from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
A CSWAP with 4T complexity and whose adjoint has 0T complexity.

                A controlled SWAP that swaps `a` and `b` based on `control`.
            It uses an extra qubit `aux` so that its adjoint would have
            a T complexity of zero.
            