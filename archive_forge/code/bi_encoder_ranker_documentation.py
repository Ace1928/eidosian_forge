from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.torch import padded_3d
from parlai.zoo.bert.build import download
from .bert_dictionary import BertDictionaryAgent
from .helpers import (
import os
import torch
from tqdm import tqdm

        Share model parameters.
        