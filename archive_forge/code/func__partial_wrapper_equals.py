from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def _partial_wrapper_equals(obs_or_fq1: _PartialWrapper, obs_or_fq2: _PartialWrapper):
    """
    Return whether the two partial wrappers are equal,
    """
    obs_or_fq1_keywords = copy.copy(obs_or_fq1.p.keywords)
    obs_or_fq2_keywords = copy.copy(obs_or_fq2.p.keywords)
    keywords_equal = True
    if 'observer' in obs_or_fq1_keywords and 'observer' in obs_or_fq2_keywords:
        keywords_equal = keywords_equal and _obs_or_fq_ctr_equals(obs_or_fq1_keywords['observer'], obs_or_fq2_keywords['observer'])
        obs_or_fq1_keywords.pop('observer')
        obs_or_fq2_keywords.pop('observer')
    keywords_equal = keywords_equal and obs_or_fq1_keywords == obs_or_fq2_keywords
    return obs_or_fq1.p.func == obs_or_fq2.p.func and obs_or_fq1.p.args == obs_or_fq2.p.args and keywords_equal