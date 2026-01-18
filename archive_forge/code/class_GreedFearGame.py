import logging
from abc import ABC
from collections import Iterable
from typing import Dict, Optional
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface
from ray.rllib.examples.env.utils.mixins import (
class GreedFearGame(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    R = 3
    P = 1
    T = R + greed
    S = P - fear
    PAYOUT_MATRIX = np.array([[[R, R], [S, T]], [[T, S], [P, P]]])
    NAME = 'IteratedGreedFear'

    def __str__(self):
        return f'{self.NAME} with greed={greed} and fear={fear}'