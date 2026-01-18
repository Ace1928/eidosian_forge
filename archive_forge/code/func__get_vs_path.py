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
def _get_vs_path(self, subdir: str) -> str:
    """
        Return a unique path for the set of comparison combos given a subdirectory.

        We hash the filename as it can grow quite large.
        """
    assert subdir
    os.makedirs(os.path.join(self.root_dir, subdir), exist_ok=True)

    def _combo_name(id1, id2):
        """
            Return joined name for combo of comparisons.
            """
        id1_name = id1
        id2_name = id2
        if 'model' in CONFIG[id1]:
            id1_name += self.task.replace(':', '_')
        if 'model' in CONFIG[id2]:
            id2_name += self.task.replace(':', '_')
        return f'{id1_name}__vs__{id2_name}'
    return os.path.join(self.root_dir, subdir, hashlib.sha1('___and___'.join([f'{_combo_name(id1, id2)}' for id1, id2 in self.combos]).encode('utf-8')).hexdigest()[:10])