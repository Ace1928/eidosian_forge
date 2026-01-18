import logging
import threading
import time
from typing import Union, Optional
from enum import Enum
import ray.cloudpickle as pickle
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import (
@PublicAPI
class Commands(Enum):
    ACTION_SPACE = 'ACTION_SPACE'
    OBSERVATION_SPACE = 'OBSERVATION_SPACE'
    GET_WORKER_ARGS = 'GET_WORKER_ARGS'
    GET_WEIGHTS = 'GET_WEIGHTS'
    REPORT_SAMPLES = 'REPORT_SAMPLES'
    START_EPISODE = 'START_EPISODE'
    GET_ACTION = 'GET_ACTION'
    LOG_ACTION = 'LOG_ACTION'
    LOG_RETURNS = 'LOG_RETURNS'
    END_EPISODE = 'END_EPISODE'