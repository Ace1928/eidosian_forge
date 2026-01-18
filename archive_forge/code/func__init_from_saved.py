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
def _init_from_saved(self, fname):
    print('[ Loading model %s ]' % fname)
    saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
    if 'word_dict' in saved_params:
        self.word_dict.copy_dict(saved_params['word_dict'])
    self.feature_dict = saved_params['feature_dict']
    self.state_dict = saved_params['state_dict']
    config.override_args(self.opt, saved_params['config'])
    self.model = DocReaderModel(self.opt, self.word_dict, self.feature_dict, self.state_dict)