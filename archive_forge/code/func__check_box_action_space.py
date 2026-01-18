import inspect
from functools import partial
from typing import Callable
import numpy as np
from gym import Space, error, logger, spaces
def _check_box_action_space(action_space: spaces.Box):
    """Checks that a :class:`Box` action space is defined in a sensible way.

    Args:
        action_space: A box action space
    """
    assert action_space.low.shape == action_space.shape, f'The Box action space shape and low shape have have different shapes, low shape: {action_space.low.shape}, box shape: {action_space.shape}'
    assert action_space.high.shape == action_space.shape, f'The Box action space shape and high shape have different shapes, high shape: {action_space.high.shape}, box shape: {action_space.shape}'
    if np.any(action_space.low == action_space.high):
        logger.warn('A Box action space maximum and minimum values are equal.')
    elif np.any(action_space.high < action_space.low):
        logger.warn('A Box action space low value is greater than a high value.')