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
@dataclass
class ModelDataConfig:
    name: str
    system: str
    role_prefix: dict
    ai_role: str
    eot_token: str
    bos_token: Optional[str]
    max_tokens: int
    pad_token: int
    ignore_id: int