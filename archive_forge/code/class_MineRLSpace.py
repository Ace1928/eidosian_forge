import random
import string
import gym
import gym.spaces
import numpy as np
import random
from collections import OrderedDict
from typing import List
import gym
import logging
import gym.spaces
import numpy as np
import collections
import warnings
import abc
class MineRLSpace(abc.ABC, gym.Space):
    """
    An interface for MineRL spaces.
    """

    @property
    def flattened(self) -> gym.spaces.Box:
        if not hasattr(self, '_flattened'):
            self._flattened = self.create_flattened_space()
        return self._flattened

    @abc.abstractmethod
    def no_op(self, batch_shape=()):
        pass

    @abc.abstractmethod
    def create_flattened_space(self):
        pass

    @abc.abstractmethod
    def flat_map(self, x):
        pass

    @abc.abstractmethod
    def unmap(self, x):
        pass

    def is_flattenable(self):
        return True

    @abc.abstractmethod
    def sample(self, bdim=None):
        pass

    def noop(self, batch_shape=()):
        """Backwards compatibility layer.

        Args:
            batch_shape (tuple, optional): [description]. Defaults to ().

        Returns:
            np.ndarray: the No_op action.
        """
        warnings.warn('space.noop() is being deprecated for space.no_op() in MineRL 1.0.0. Please change your code to reflect this change.', DeprecationWarning)
        return self.no_op(batch_shape)