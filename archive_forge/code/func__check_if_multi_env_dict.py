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
def _check_if_multi_env_dict(env, element, function_string):
    if not isinstance(element, dict):
        raise ValueError(f'The element returned by {function_string} is not a MultiEnvDict. Instead, it is of type: {type(element)}')
    env_ids = env.get_sub_environments(as_dict=True).keys()
    if not all((k in env_ids for k in element)):
        raise ValueError(f"The element returned by {function_string} has dict keys that don't correspond to environment ids for this env {list(env_ids)}")
    for _, multi_agent_dict in element.items():
        _check_if_element_multi_agent_dict(env, multi_agent_dict, function_string, base_env=True)