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
def _init_from_scratch(self):
    self.feature_dict = build_feature_dict(self.opt)
    self.opt['num_features'] = len(self.feature_dict)
    self.opt['vocab_size'] = len(self.word_dict)
    print('[ Initializing model from scratch ]')
    self.model = DocReaderModel(self.opt, self.word_dict, self.feature_dict)
    self.model.set_embeddings()