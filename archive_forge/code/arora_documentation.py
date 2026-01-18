from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.build_data import modelzoo_path
import torchtext.vocab as vocab
from parlai.utils.misc import TimeLogger
from collections import Counter, deque
import numpy as np
import os
import pickle
import torch

        Produce a Arora-style sentence embedding for a given sentence.

        Inputs:
          sent: tokenized sentence; a list of strings
          rem_first_sv: If True, remove the first singular value when you compute the
            sentence embddings. Otherwise, don't remove it.
        Returns:
          sent_emb: tensor length glove_dim, or None.
              If sent_emb is None, that's because all of the words were OOV for GloVe.
        