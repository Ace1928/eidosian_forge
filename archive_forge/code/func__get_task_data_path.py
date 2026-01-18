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
def _get_task_data_path(self, config_id: str) -> str:
    """
        Return path to task data as conversations for given task.

        :param config_id:
            config_id string
        """
    task_data_dir = os.path.join(self.root_dir, 'tasks_as_conversations')
    os.makedirs(task_data_dir, exist_ok=True)
    return os.path.join(task_data_dir, f'{config_id}.jsonl')