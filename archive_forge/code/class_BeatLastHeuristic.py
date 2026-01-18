import gymnasium as gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.view_requirement import ViewRequirement
class BeatLastHeuristic(Policy):
    """Play the move that would beat the last move of the opponent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):

        def successor(x):
            if isinstance(self.observation_space, gym.spaces.Discrete):
                if x == ROCK:
                    return PAPER
                elif x == PAPER:
                    return SCISSORS
                elif x == SCISSORS:
                    return ROCK
                else:
                    return random.choice([ROCK, PAPER, SCISSORS])
            elif x[ROCK] == 1:
                return PAPER
            elif x[PAPER] == 1:
                return SCISSORS
            elif x[SCISSORS] == 1:
                return ROCK
            elif x[-1] == 1:
                return random.choice([ROCK, PAPER, SCISSORS])
        return ([successor(x) for x in obs_batch], [], {})

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass