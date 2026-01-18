import json
from typing import Union
import gym
from minerl.herobraine.env_spec import EnvSpec
def _gym_space_to_dict(space: Union[dict, gym.Space]) -> dict:
    if isinstance(space, gym.spaces.Dict):
        dct = {}
        for k in space.spaces:
            dct[k] = _gym_space_to_dict(space.spaces[k])
        return dct
    else:
        return space