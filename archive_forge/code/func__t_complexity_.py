from typing import Iterable, Sequence, Union
import attr
import cirq
import numpy as np
from cirq_ft import infra
from cirq_ft.deprecation import deprecated_cirq_ft_class
def _t_complexity_(self) -> infra.TComplexity:
    return infra.TComplexity(t=self.target_bitsize)