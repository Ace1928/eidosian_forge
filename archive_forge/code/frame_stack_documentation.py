from collections import deque
from typing import Union
import numpy as np
import gym
from gym.error import DependencyNotInstalled
from gym.spaces import Box
Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        