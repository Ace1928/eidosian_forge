import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def _one_unique(df, name, default):
    """Helper function to assert that there's one unique value in a column and return it."""
    if name not in df.columns:
        return default
    vals = df[name].unique()
    assert len(vals) == 1, name
    return vals[0]