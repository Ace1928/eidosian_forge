import gymnasium as gym
from ray.rllib.utils.annotations import PublicAPI
Gym Dictionary with arbitrary keys updatable after instantiation

    Example:
       space = FlexDict({})
       space['key'] = spaces.Box(4,)
    See also: documentation for gym.spaces.Dict
    