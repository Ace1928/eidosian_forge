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
class MultiDiscrete(gym.spaces.MultiDiscrete, MineRLSpace):

    def __init__(self, *args, **kwargs):
        super(MultiDiscrete, self).__init__(*args, **kwargs)
        self.eyes = [np.eye(n, dtype=np.float32) for n in self.nvec]

    def no_op(self, batch_shape=()):
        return (np.zeros(list(batch_shape) + list(self.nvec.shape)) * self.nvec).astype(self.dtype)

    def create_flattened_space(self):
        return Box(low=0, high=1, shape=[np.sum(self.nvec)])

    def flat_map(self, x):
        return np.concatenate([self.eyes[i][x[..., i]] for i in range(len(self.nvec))], axis=-1)

    def unmap(self, x):
        cur_index = 0
        out = []
        for n in self.nvec:
            out.append(np.argmax(x[..., cur_index:cur_index + n], axis=-1)[..., np.newaxis])
            cur_index += n
        return np.concatenate(out, axis=-1).astype(self.dtype)

    def sample(self, bs=None):
        bdim = () if bs is None else (bs,)
        return (self.np_random.random_sample(bdim + self.nvec.shape) * self.nvec).astype(self.dtype)