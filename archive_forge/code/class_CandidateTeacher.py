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
class CandidateTeacher(CandidateBaseTeacher, DialogTeacher):
    """
    Candidate teacher produces several candidates, one of which is a repeat of the
    input.

    A good ranker should easily identify the correct response.
    """

    def setup_data(self, fold):
        super()._setup_data(fold)
        for i, text in enumerate(self.corpus):
            cands = []
            for j in range(NUM_CANDIDATES):
                offset = (i + j) % len(self.corpus)
                cands.append(self.corpus[offset])
            yield ((text, [text], 0, cands), True)