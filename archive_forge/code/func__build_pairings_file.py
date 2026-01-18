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
def _build_pairings_file(self):
    """
        Build and save pairings to pairings file.
        """
    onboarding_pairs = []
    with open(self.onboarding_path) as f:
        for line in f:
            onboarding_pairs.append(json.loads(line))
    pairings_filepath = self._get_vs_path('pairings_files')
    self._print_progress(f'building pairings file, saving at {pairings_filepath}')
    conversations = {config_id: Conversations(self.chat_files[config_id]) for config_id in self.config_ids}
    pairs = self._build_conversation_pairs(conversations)
    with open(pairings_filepath, 'w') as f:
        pairs = onboarding_pairs + pairs
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')