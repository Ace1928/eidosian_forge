from typing import NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from gym.logger import warn
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.space import Space
def _generate_sample_space(self, base_space: Union[None, Box, Discrete], num: int) -> Optional[Union[Box, MultiDiscrete]]:
    if num == 0 or base_space is None:
        return None
    if isinstance(base_space, Box):
        return Box(low=np.array(max(1, num) * [base_space.low]), high=np.array(max(1, num) * [base_space.high]), shape=(num,) + base_space.shape, dtype=base_space.dtype, seed=self.np_random)
    elif isinstance(base_space, Discrete):
        return MultiDiscrete(nvec=[base_space.n] * num, seed=self.np_random)
    else:
        raise TypeError(f'Expects base space to be Box and Discrete, actual space: {type(base_space)}.')