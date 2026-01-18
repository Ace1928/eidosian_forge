import numpy as np
import gym
from gym import ActionWrapper
from gym.spaces import Box
Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        