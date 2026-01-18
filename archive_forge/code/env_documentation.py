import logging
import traceback
from copy import copy
from typing import TYPE_CHECKING, Optional, Set, Union
import numpy as np
import tree  # pip install dm_tree
from ray.actor import ActorHandle
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.error import ERR_MSG_OLD_GYM_API, UnsupportedSpaceException
from ray.rllib.utils.gym import check_old_gym_env, try_import_gymnasium_and_gym
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import EnvType
from ray.util import log_once
Returns error, value, and space when offending `space.contains(value)` fails.

    Returns only the offending sub-value/sub-space in case `space` is a complex Tuple
    or Dict space.

    Args:
        space: The gym.Space to check.
        value: The actual (numpy) value to check for matching `space`.

    Returns:
        Tuple consisting of 1) key-sequence of the offending sub-space or the empty
        string if `space` is not complex (Tuple or Dict), 2) the offending sub-space,
        3) the offending sub-space's dtype, 4) the offending sub-value, 5) the offending
        sub-value's dtype.

    .. testcode::
        :skipif: True

        path, space, space_dtype, value, value_dtype = _find_offending_sub_space(
            gym.spaces.Dict({
           -2.0, 1.5, (2, ), np.int8), np.array([-1.5, 3.0])
        )

    