from typing import Optional
from dataclasses import dataclass
import argparse
import json
import os
import random
import numpy as np
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from ray.util.multiprocessing import Pool
def _convert_single_conversation(c):
    tokens = []
    masks = []
    if CONFIG.bos_token:
        t = tokenizer.convert_tokens_to_ids(CONFIG.bos_token)
        tokens.append(t)
        masks.append(False)
    if CONFIG.system:
        t = tokenizer(CONFIG.system, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids(CONFIG.eot_token)]
        tokens.extend(t)
        masks.extend([False] * len(t))
    for message in c['items']:
        message_text = CONFIG.role_prefix[message['from']] + message['value']
        t = tokenizer(message_text, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids(CONFIG.eot_token)]
        tokens.extend(t)
        masks.extend([message['from'] == CONFIG.ai_role] * len(t))
    return (tokens, masks)