from parlai import __file__ as parlai_filepath
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.acute_eval.run import AcuteEvaluator, add_args as acute_add_args
from parlai.scripts.self_chat import self_chat, setup_args as self_chat_setup_args
from parlai.utils.conversations import Conversations, Conversation
from parlai.utils.strings import normalize_reply
from parlai.utils.testing import capture_output
from parlai.mturk.tasks.acute_eval.analysis import (
from parlai.mturk.tasks.acute_eval.dump_task_to_acute_format import (
from parlai.mturk.tasks.acute_eval.configs import CONFIG
from typing import Dict, Any, List, Tuple, Set
from itertools import combinations
import datetime
import time
import json
import os
import random
import torch
import hashlib
def _get_log_path(self, config_id: str) -> str:
    """
        Return path to chat logs given config_id.

        :param identifier:
            config_id in CONFIG.
        """
    config = CONFIG[config_id]
    path = ''
    if 'log_path' in config:
        path = config['log_path']
        assert os.path.exists(path), f'Path provided in log_path for {config_id} does not exist'
    elif 'task' in config:
        path = self._get_task_data_path(config_id)
    elif 'model' in config:
        path = self._get_selfchat_log_path(config_id)
    assert path, f'Invalid config for {config_id}'
    return path