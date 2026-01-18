from typing import Tuple
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import select_and_prepare
from cirq_ft.algos.mean_estimation import arctan
Applies $ROT_{y}|l>|garbage_{l}> = exp(i * -2arctan{y_{l}})|l>|garbage_{l}>$.

    TODO(#6142): This currently assumes that the random variable `y_{l}` only takes integer
    values. This constraint can be removed by using a standardized floating point to
    binary encoding, like IEEE 754, to encode arbitrary floats in the binary target
    register and use them to compute the more accurate $-2arctan{y_{l}}$ for any arbitrary
    $y_{l}$.
    