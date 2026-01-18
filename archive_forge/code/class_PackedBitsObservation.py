import gym
import numpy as np
import tensorflow as tf
class PackedBitsObservation(gym.ObservationWrapper):
    """Wrapper that encodes a frame as packed bits instead of booleans.

  8x less to be transferred across the wire (16 booleans stored as uint16
  instead of 16 uint8) and 8x less to be transferred from CPU to TPU (16
  booleans stored as uint32 instead of 16 bfloat16).

  """

    def __init__(self, env):
        super(PackedBitsObservation, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=np.iinfo(np.uint16).max, shape=env.observation_space.shape[:-1] + ((env.observation_space.shape[-1] + 15) // 16,), dtype=np.uint16)

    def observation(self, observation):
        data = np.packbits(observation, axis=-1)
        if data.shape[-1] % 2 == 1:
            data = np.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, 1)], 'constant')
        return data.view(np.uint16)