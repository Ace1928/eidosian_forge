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
class MultiturnCandidateTeacher(CandidateTeacher):
    """
    Splits inputs/targets by spaces into multiple turns.

    Good for testing models that use the dialog history.
    """

    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, a, r, cs), _e in raw:
            split_t = t.split(' ')
            split_a = a[0].split(' ')
            split_cs = [c.split(' ') for c in cs]
            for i in range(len(split_t)):
                yield ((split_t[i], [' '.join(split_a[:i + 1])], r, [' '.join(c[:i + 1]) for c in split_cs]), i == 0)

    def num_examples(self):
        return self.example_size * self.num_episodes()