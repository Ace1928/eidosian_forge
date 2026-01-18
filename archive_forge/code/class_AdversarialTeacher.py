import parlai.core.build_data as build_data
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from .base_agent import _BaseSafetyTeacher
from .build import build
import copy
import json
import os
import random
import sys as _sys
class AdversarialTeacher(_BaseSafetyTeacher):
    """
    Data from the adversarial collection described in the paper `Build it Break it Fix
    it for Dialogue Safety: Robustness from Adversarial Human Attack`
    (<https://arxiv.org/abs/1908.06083>)

    To see data from rounds 1, 2, and 3, try running:
    `parlai display_data -t dialogue_safety:adversarial --round 3`

    To see data from round 2 only, try running:
    `parlai display_data -t dialogue_safety:adversarial --round 2
     --round-only True`
    """

    def _load_data_dump(self):
        with open(self.data_path, 'rb') as f:
            dump = json.load(f)
        return dump['adversarial']