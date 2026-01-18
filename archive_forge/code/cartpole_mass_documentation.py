from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

        Returns:
            Tuple[float]: The current mass of the cart- and pole.
        