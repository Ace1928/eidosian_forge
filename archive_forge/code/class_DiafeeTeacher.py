import copy
import json
import os
import re
import numpy as np
from parlai.core.teachers import ParlAIDialogTeacher, MultiTaskTeacher
from projects.self_feeding.utils import add_person_tokens
from .build import build
class DiafeeTeacher(SelfFeedingMTLTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtasks'] = ['dialog', 'feedback']
        train_files = [opt['dia_train'], opt['fee_train']]
        assert len(opt['subtasks']) == len(train_files)
        tasks = [f'self_feeding:{subtask}:{train_file}' for subtask, train_file in zip(opt['subtasks'], train_files)]
        opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)