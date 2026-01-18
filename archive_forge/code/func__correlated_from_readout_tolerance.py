import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def _correlated_from_readout_tolerance(readout_tolerance: float) -> float:
    """Heuristic formula for the off-diagonal confusion matrix error thresholds.

    This is chosen to return 0.3 for readout_tolerance = 0.4 and 1.0 for readout_tolerance = 1.0.
    """
    return max(0.0, min(1.0, 7 / 6 * readout_tolerance - 1 / 6))