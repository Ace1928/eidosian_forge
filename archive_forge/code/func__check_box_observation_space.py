import inspect
from functools import partial
from typing import Callable
import numpy as np
from gym import Space, error, logger, spaces
def _check_box_observation_space(observation_space: spaces.Box):
    """Checks that a :class:`Box` observation space is defined in a sensible way.

    Args:
        observation_space: A box observation space
    """
    if len(observation_space.shape) == 3:
        if observation_space.dtype != np.uint8:
            logger.warn(f'It seems a Box observation space is an image but the `dtype` is not `np.uint8`, actual type: {observation_space.dtype}. If the Box observation space is not an image, we recommend flattening the observation to have only a 1D vector.')
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            logger.warn('It seems a Box observation space is an image but the upper and lower bounds are not in [0, 255]. Generally, CNN policies assume observations are within that range, so you may encounter an issue if the observation values are not.')
    if len(observation_space.shape) not in [1, 3]:
        logger.warn(f'A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: {observation_space.shape}')
    assert observation_space.low.shape == observation_space.shape, f'The Box observation space shape and low shape have different shapes, low shape: {observation_space.low.shape}, box shape: {observation_space.shape}'
    assert observation_space.high.shape == observation_space.shape, f'The Box observation space shape and high shape have have different shapes, high shape: {observation_space.high.shape}, box shape: {observation_space.shape}'
    if np.any(observation_space.low == observation_space.high):
        logger.warn('A Box observation space maximum and minimum values are equal.')
    elif np.any(observation_space.high < observation_space.low):
        logger.warn('A Box observation space low value is greater than a high value.')