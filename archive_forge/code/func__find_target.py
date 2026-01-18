import bisect
import os
import numpy as np
import json
import random
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.build_data import modelzoo_path
from . import config
from .utils import build_feature_dict, vectorize, batchify, normalize_text
from .model import DocReaderModel
def _find_target(self, document, labels):
    """
        Find the start/end token span for all labels in document.

        Return a random one for training.
        """

    def _positions(d, l):
        for i in range(len(d)):
            for j in range(i, min(len(d) - 1, i + len(l))):
                if l == d[i:j + 1]:
                    yield (i, j)
    targets = []
    for label in labels:
        targets.extend(_positions(document, self.word_dict.tokenize(label)))
    if len(targets) == 0:
        return
    return targets[np.random.choice(len(targets))]