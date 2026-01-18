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
def _check_done_and_truncated(done, truncated, base_env=False, agent_ids=None):
    for what in ['done', 'truncated']:
        data = done if what == 'done' else truncated
        if base_env:
            for _, multi_agent_dict in data.items():
                for agent_id, done_ in multi_agent_dict.items():
                    if not isinstance(done_, (bool, np.bool_)):
                        raise ValueError(f'Your step function must return `{what}s` that are boolean. But instead was a {type(data)}')
                    if not (agent_id in agent_ids or agent_id == '__all__'):
                        error = f'Your `{what}s` dictionary must have agent ids that belong to the environment. Agent_ids recieved from env.get_agent_ids() are: {agent_ids}'
                        raise ValueError(error)
        elif not isinstance(data, (bool, np.bool_)):
            error = f'Your step function must return a `{what}` that is a boolean. But instead was a {type(data)}'
            raise ValueError(error)