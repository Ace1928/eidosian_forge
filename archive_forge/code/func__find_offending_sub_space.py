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
def _find_offending_sub_space(space, value):
    """Returns error, value, and space when offending `space.contains(value)` fails.

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

    """
    if not isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple)):
        return (None, space, space.dtype, value, _get_type(value))
    structured_space = get_base_struct_from_space(space)

    def map_fn(p, s, v):
        if not s.contains(v):
            raise UnsupportedSpaceException((p, s, v))
    try:
        tree.map_structure_with_path(map_fn, structured_space, value)
    except UnsupportedSpaceException as e:
        space, value = (e.args[0][1], e.args[0][2])
        return ('->'.join(e.args[0][0]), space, space.dtype, value, _get_type(value))
    return (None, None, None, None, None)