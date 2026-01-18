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
def _balance_data(self, data_list):
    ok = [x for x in data_list if x['is_sensitive'] == 0]
    notok = [x for x in data_list if x['is_sensitive'] == 1]
    new_not_ok = []
    while len(new_not_ok) < len(ok):
        new_not_ok.append(self.fixed_random.choice(notok))
    new_data = ok + new_not_ok
    self.fixed_random.shuffle(new_data)
    return new_data