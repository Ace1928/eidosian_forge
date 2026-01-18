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
def _get_unique_ids(self, conversations: Dict[str, Conversations]) -> Dict[str, List[int]]:
    """
        Assign unique IDs for each conversation in conversations.

        This is important for ACUTE-Eval, since we do not want evaluators
        to see the same conversations across comparisons.

        :param conversations:
            Dict mapping config ID to list of conversations

        :return unique_ids:
            dict mapping config id to list of conversation IDs
        """
    id_num = 0
    unique_ids: Dict[str, List[int]] = {}
    for config_id in self.config_ids:
        unique_ids[config_id] = []
        for _convo in conversations[config_id]:
            unique_ids[config_id].append(id_num)
            id_num += 1
    return unique_ids