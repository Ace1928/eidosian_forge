from copy import deepcopy
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
Wrapper for gym CartPole environment where reward is accumulated to the end.