import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def _iter_angles_for_characterization(self) -> Iterable[Tuple[Optional[float], 'sympy.Symbol']]:
    yield from ((default, symbol) for characterize, default, symbol in self._iter_angles() if characterize)