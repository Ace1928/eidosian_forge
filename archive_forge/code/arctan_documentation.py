from typing import Iterable, Sequence, Union
import attr
import cirq
import numpy as np
from cirq_ft import infra
from cirq_ft.deprecation import deprecated_cirq_ft_class
Applies U|x>|0>|0000...0> = |x>|sign>|abs(-2 arctan(x) / pi)>.

    Args:
        selection_bitsize: The bitsize of input register |x>.
        target_bitsize: The bitsize of output register. The computed quantity,
            $\abs(-2 * \arctan(x) / \pi)$ is stored as a fixed-length binary approximation
            in the output register of size `target_bitsize`.
    