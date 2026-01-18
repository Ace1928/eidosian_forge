import gymnasium as gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.view_requirement import ViewRequirement
class AlwaysSameHeuristic(Policy):
    """Pick a random move and stick with it for the entire episode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()
        self.view_requirements.update({'state_in_0': ViewRequirement('state_out_0', shift=-1, space=gym.spaces.Box(ROCK, SCISSORS, shape=(1,), dtype=np.int32))})

    def get_initial_state(self):
        return [random.choice([ROCK, PAPER, SCISSORS])]

    def is_recurrent(self) -> bool:
        return True

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        return ([state_batches[0][0] for x in obs_batch], state_batches, {})