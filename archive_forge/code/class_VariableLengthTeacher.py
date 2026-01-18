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
class VariableLengthTeacher(CandidateTeacher):

    def build_corpus(self):
        corpus = super().build_corpus()
        for i in range(len(corpus)):
            length = len(corpus[i]) - i % 3
            corpus[i] = corpus[i][:length]
        return corpus