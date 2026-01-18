import sys
from typing import (
import numpy as np
from gym import spaces
from gym.logger import warn
from gym.utils import seeding
class ObservationWrapper(Wrapper):
    """Superclass of wrappers that can modify observations using :meth:`observation` for :meth:`reset` and :meth:`step`.

    If you would like to apply a function to the observation that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`ObservationWrapper` and overwrite the method
    :meth:`observation` to implement that transformation. The transformation defined in that method must be
    defined on the base environment’s observation space. However, it may take values in a different space.
    In that case, you need to specify the new observation space of the wrapper by setting :attr:`self.observation_space`
    in the :meth:`__init__` method of your wrapper.

    For example, you might have a 2D navigation task where the environment returns dictionaries as observations with
    keys ``"agent_position"`` and ``"target_position"``. A common thing to do might be to throw away some degrees of
    freedom and only consider the position of the target relative to the agent, i.e.
    ``observation["target_position"] - observation["agent_position"]``. For this, you could implement an
    observation wrapper like this::

        class RelativePosition(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

            def observation(self, obs):
                return obs["target"] - obs["agent"]

    Among others, Gym provides the observation wrapper :class:`TimeAwareObservation`, which adds information about the
    index of the timestep to the observation.
    """

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(**kwargs)
        return (self.observation(obs), info)

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (self.observation(observation), reward, terminated, truncated, info)

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError