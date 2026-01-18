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
class BadExampleTeacher(CandidateTeacher):
    """
    Teacher which produces a variety of examples that upset verify_data.py.

    Useful for checking how models respond when the following assumptions are
    violated:

        0. text is empty string
        1. missing text
        2. label is empty string
        3. missing label
        4. label candidates is empty
        5. label candidates contains an empty string
        6. label isn't in the candidates
        7. missing label candidates

    Note: this test may come to outlive its purpose in the future. When failing
    this test, one should consider who is really at fault: the test, or the code.
    """
    NUM_CASES = 8

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.data.get = self._wrapperfn(self.data.get)

    def _wrapperfn(self, oldget):

        def newget(*args):
            item, eod = oldget(*args)
            item = copy.deepcopy(item)
            newget.case = (newget.case + 1) % self.NUM_CASES
            case = newget.case
            if case == 0:
                item.force_set('text', '')
            elif case == 1:
                del item['text']
            elif case == 2:
                item.force_set('labels', [''])
            elif case == 3:
                del item['labels']
            elif case == 4:
                item.force_set('label_candidates', [])
            elif case == 5:
                item.force_set('label_candidates', list(item['label_candidates']) + [''])
            elif case == 6:
                item.force_set('label_candidates', list(item['label_candidates']))
                item['label_candidates'].remove(item['labels'][0])
            elif case == 7:
                del item['label_candidates']
            return (item, eod)
        newget.case = random.randint(0, self.NUM_CASES)
        return newget