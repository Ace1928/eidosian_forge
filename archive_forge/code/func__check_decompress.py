from collections import deque
from typing import Union
import numpy as np
import gym
from gym.error import DependencyNotInstalled
from gym.spaces import Box
def _check_decompress(self, frame):
    if self.lz4_compress:
        from lz4.block import decompress
        return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(self.frame_shape)
    return frame