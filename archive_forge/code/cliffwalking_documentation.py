from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import numpy as np
from gym import Env, logger, spaces
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
Determine the outcome for an action. Transition Prob is always 1.0.

        Args:
            current: Current position on the grid as (row, col)
            delta: Change in position for transition

        Returns:
            Tuple of ``(1.0, new_state, reward, terminated)``
        