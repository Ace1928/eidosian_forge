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
class DiscreteRange(Discrete):
    """
    {begin, begin+1, ..., end-2, end - 1}
    
    Like discrete, but takes a range of dudes
    DiscreteRange(0, n) is equivalent to Discrete(n)

    Examples usage:
    self.observation_space = spaces.DiscreteRange(-1, 3)
    """

    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        super().__init__(end - begin)

    def sample(self, bs=None):
        return super().sample(bs) + self.begin

    def contains(self, x):
        return super().contains(x - self.begin)
    __contains__ = contains

    def no_op(self, batch_shape=()):
        if len(batch_shape) == 0:
            return self.begin
        else:
            return (np.zeros(batch_shape) + self.begin).astype(self.dtype)

    def create_flattened_space(self):
        return Box(low=0, high=1, shape=(self.n,))

    def flat_map(self, x):
        return self.eye[x - self.begin]

    def unmap(self, x):
        return np.array(np.argmax(x, axis=-1) + self.begin, dtype=self.dtype)

    def __repr__(self):
        return 'DiscreteRange({}, {})'.format(self.begin, self.n + self.begin)

    def __eq__(self, other):
        return self.n == other.n and self.begin == other.begin