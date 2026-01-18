from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
class RepeatTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = 'unused_path'
        task = opt.get('task', 'integration_tests:RepeatTeacher:50')
        try:
            self.data_length = int(task.split(':')[-1])
        except ValueError:
            self.data_length = 10
        super().__init__(opt, shared)

    def setup_data(self, unused_path):
        for i in range(self.data_length):
            yield ((str(i), [str(i)]), True)

    def num_examples(self):
        return self.data_length

    def num_episodes(self):
        return self.data_length