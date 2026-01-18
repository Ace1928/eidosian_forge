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
class MultiturnTeacher(FixedDialogTeacher):
    """
    Data from the multi-turn adversarial collection described in the paper `Build it
    Break it Fix it for Dialogue Safety: Robustness from Adversarial Human Attack`
    (<https://arxiv.org/abs/1908.06083>)

    To see data containing multi-turn conversations, try running
    `parlai display_data -t dialogue_safety:multiturn`.

    Run the above command with the flag `--single-turn True` to only see the
    single turn data.
    """

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('Multiturn Safety Teacher Args')
        parser.add_argument('--single-turn', type='bool', default=False, help='only include the single turn data and not the context info')

    def __init__(self, opt, shared=None):
        build(opt['datapath'])
        self.opt = opt
        self.data_path = os.path.join(opt['datapath'], 'dialogue_safety', MULTI_TURN_DATA)
        self.fixed_random = random.Random(42)
        self.single_turn = opt['single_turn']
        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self._setup_data(opt['datatype'])
        super().__init__(opt, shared)
        self.reset()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared

    def _setup_data(self, datatype):
        dt = datatype.split(':')[0]
        self.all_data = json.load(open(self.data_path, 'rb'))
        data = self.all_data[dt]
        if self.single_turn:
            new_data = []
            for datum in data:
                datum['text'] = datum['text'].split('\n')[-1]
                new_data.append(datum)
            self.data = new_data
        else:
            self.data = data

    def get(self, episode_idx, entry_idx):
        return Message(self.data[episode_idx])