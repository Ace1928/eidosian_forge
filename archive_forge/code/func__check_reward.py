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
def _check_reward(reward, base_env=False, agent_ids=None):
    if base_env:
        for _, multi_agent_dict in reward.items():
            for agent_id, rew in multi_agent_dict.items():
                if not (np.isreal(rew) and (not isinstance(rew, bool)) and (np.isscalar(rew) or (isinstance(rew, np.ndarray) and rew.shape == ()))):
                    error = f'Your step function must return rewards that are integer or float. reward: {rew}. Instead it was a {type(rew)}'
                    raise ValueError(error)
                if not (agent_id in agent_ids or agent_id == '__all__'):
                    error = f'Your reward dictionary must have agent ids that belong to the environment. Agent_ids recieved from env.get_agent_ids() are: {agent_ids}'
                    raise ValueError(error)
    elif not (np.isreal(reward) and (not isinstance(reward, bool)) and (np.isscalar(reward) or (isinstance(reward, np.ndarray) and reward.shape == ()))):
        error = 'Your step function must return a reward that is integer or float. Instead it was a {}'.format(type(reward))
        raise ValueError(error)