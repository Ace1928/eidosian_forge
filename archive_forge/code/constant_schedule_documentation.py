from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.schedules.schedule import Schedule
from ray.rllib.utils.typing import TensorType
Initializes a ConstantSchedule instance.

        Args:
            value: The constant value to return, independently of time.
            framework: The framework descriptor string, e.g. "tf",
                "torch", or None.
        