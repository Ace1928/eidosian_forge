import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
def _setting_to_z_observable(setting: InitObsSetting):
    qubits = setting.observable.qubits
    return InitObsSetting(init_state=zeros_state(qubits), observable=ops.PauliString(qubit_pauli_map={q: ops.Z for q in qubits}))