from typing import Optional
import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.utils import try_import_pyspiel
def _solve_chance_nodes(self):
    while self.state.is_chance_node():
        assert self.state.current_player() == -1
        actions, probs = zip(*self.state.chance_outcomes())
        action = np.random.choice(actions, p=probs)
        self.state.apply_action(action)