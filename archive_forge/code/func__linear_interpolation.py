from typing import Callable, List, Optional, Tuple
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType
def _linear_interpolation(left, right, alpha):
    return left + alpha * (right - left)