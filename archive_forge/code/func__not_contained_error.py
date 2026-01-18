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
def _not_contained_error(func_name, _type):
    _error = f"The {_type} collected from {func_name} was not contained within your env's {_type} space. Its possible that there was a typemismatch (for example {_type}s of np.float32 and a space ofnp.float64 {_type}s), or that one of the sub-{_type}s wasout of bounds"
    return _error