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
class ShortFixedTeacher(FixedDialogCandidateTeacher):
    """
    Fixed Dialog Candidate teacher with only 10 training examples.
    """

    def __init__(self, opt: Opt, shared: dict=None):
        super().__init__(opt, shared, num_train=10, num_test=10)