import sys
from typing import Any, Dict, Optional, Tuple
import gym
from gym.core import ObsType
from gym.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
@runtime_checkable
class LegacyEnv(Protocol):
    """A protocol for environments using the old step API."""
    observation_space: gym.Space
    action_space: gym.Space

    def reset(self) -> Any:
        """Reset the environment and return the initial observation."""
        ...

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Run one timestep of the environment's dynamics."""
        ...

    def render(self, mode: Optional[str]='human') -> Any:
        """Render the environment."""
        ...

    def close(self):
        """Close the environment."""
        ...

    def seed(self, seed: Optional[int]=None):
        """Set the seed for this env's random number generator(s)."""
        ...