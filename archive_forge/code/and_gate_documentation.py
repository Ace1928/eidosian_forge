from typing import Sequence, Tuple
import numpy as np
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls- 2.