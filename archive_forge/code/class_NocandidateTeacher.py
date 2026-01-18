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
class NocandidateTeacher(CandidateTeacher):
    """
    Strips the candidates so the model can't see any options.

    Good for testing simple generative models.
    """

    def setup_data(self, fold):
        raw = super().setup_data(fold)
        for (t, a, _r, _c), e in raw:
            yield ((t, a), e)