import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
def _validate_setting(self, setting: InitObsSetting, what: str):
    mws = _max_weight_state([self.max_setting.init_state, setting.init_state])
    mwo = _max_weight_observable([self.max_setting.observable, setting.observable])
    if mws is None or mwo is None:
        raise ValueError(f"You requested the {what} for a setting that is not compatible with this BitstringAccumulator's meas_spec.")