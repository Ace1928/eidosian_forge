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
def _build_conversation_pairs(self, conversations: Dict[str, Conversations]) -> List[Dict[str, Any]]:
    """
        Build a conversation pair to show during ACUTE Eval.

        We build twice as many pairs per matchup as specified
        in the config, to account for issues where sometimes
        we run out of pairs of conversations to evaluate.

        :param conversations:
            A dictionary mapping config_id to dialogues

        :return pairs:
            A list of conversation pairs
        """
    unique_ids = self._get_unique_ids(conversations)
    pairs = []
    pairs_per_id = self.opt['matchups_per_pair'] * 2
    for id_pair in self.combos:
        for _ in range(pairs_per_id):
            conversation_indices = [random.choice(range(len(conversations[id_]))) for id_ in id_pair]
            pair = []
            pair_ids = []
            for i, c_id in enumerate(conversation_indices):
                id_ = id_pair[i]
                pair.append(self._acutify_convo(conversations[id_][c_id], id_))
                pair_ids.append(unique_ids[id_][c_id])
            pairs.append({'is_onboarding': False, 'speakers_to_eval': id_pair, 'dialogue_dicts': pair, 'dialogue_ids': pair_ids})
    return pairs