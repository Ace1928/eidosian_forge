from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.light_dialog.agents import DefaultTeacher as OrigLightTeacher
from parlai.tasks.light_genderation_bias.build import build
from collections import deque
from copy import deepcopy
import csv
import json
import os
import random
def _get_new_data(self, opt):
    """
        Load extra positive dialogue data IFF datatype==train.
        """
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        with open(os.path.join(_path(opt), NEW_DATA), 'r') as f:
            data = json.load(f)
        new_data = []
        for ep in data:
            new_ep = []
            for ex in ep:
                ex['new_data'] = True
                new_ep.append(Message(ex))
            new_data.append(new_ep)
        return new_data
    return []