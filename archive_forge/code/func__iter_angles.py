import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def _iter_angles(self) -> Iterable[Tuple[bool, Optional[float], 'sympy.Symbol']]:
    yield from ((self.characterize_theta, self.theta_default, THETA_SYMBOL), (self.characterize_zeta, self.zeta_default, ZETA_SYMBOL), (self.characterize_chi, self.chi_default, CHI_SYMBOL), (self.characterize_gamma, self.gamma_default, GAMMA_SYMBOL), (self.characterize_phi, self.phi_default, PHI_SYMBOL))