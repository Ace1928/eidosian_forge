import logging
from typing import Callable, Tuple, Optional, List, Dict, Any, TYPE_CHECKING, Union, Set
import gymnasium as gym
import ray
from ray.rllib.utils.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
Check if the given space contains the observations of x.

        Args:
            space: The space to if x's observations are contained in.
            x: The observations to check.

        Returns:
            True if the observations of x are contained in space.
        