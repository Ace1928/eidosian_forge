import enum
from typing import Any, List, Optional, TYPE_CHECKING, Union
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from cirq import circuits, ops, study, value
from cirq._compat import proper_repr
def _create_tabulation(measurements: pd.DataFrame) -> pd.DataFrame:
    """Returns a sum of 0 and 1 results per index from a list of measurements."""
    if 'num_pulses' in measurements.columns:
        cols = [measurements.delay_ns, measurements.num_pulses]
    else:
        cols = [measurements.delay_ns]
    tabulation = pd.crosstab(cols, measurements.output).reset_index()
    for col_index, name in [(1, 0), (2, 1)]:
        if name not in tabulation:
            tabulation.insert(col_index, name, [0] * tabulation.shape[0])
    return tabulation