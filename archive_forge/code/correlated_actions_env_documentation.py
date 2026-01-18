import gymnasium as gym
from gymnasium.spaces import Discrete, Tuple
import random

    Simple env in which the policy has to emit a tuple of equal actions.

    In each step, the agent observes a random number (0 or 1) and has to choose
    two actions a1 and a2.
    It gets +5 reward for matching a1 to the random obs and +5 for matching a2
    to a1. I.e., +10 at most per step.

    One way to effectively learn this is through correlated action
    distributions, e.g., in examples/autoregressive_action_dist.py

    There are 20 steps. Hence, the best score would be ~200 reward.
    