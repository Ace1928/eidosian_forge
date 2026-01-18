import time
from enum import Enum
from typing import Dict, Tuple, Union
from ray.util import PublicAPI
from ray.util.annotations import DeveloperAPI
@PublicAPI
class Verbosity(Enum):
    V0_MINIMAL = 0
    V1_EXPERIMENT = 1
    V2_TRIAL_NORM = 2
    V3_TRIAL_DETAILS = 3

    def __int__(self):
        return self.value