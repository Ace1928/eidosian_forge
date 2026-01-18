from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from triad import SerializableRLock
from tune._utils.math import adjust_high
from tune.concepts.flow import Trial, TrialReport
from tune.concepts.logger import make_logger
from tune.concepts.space import (
from tune.noniterative.objective import (
def _convert_transition_choice(k: str, v: TransitionChoice) -> Any:
    return (hp.randint(k, 0, len(v.values)), lambda x: v.values[int(np.round(x))])