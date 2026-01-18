from parlai.utils.torch import concat_without_padding
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.zoo.bert.build import download
from .bert_dictionary import BertDictionaryAgent
from .helpers import (
import os

    TorchRankerAgent implementation of the crossencoder.

    It is a standalone Agent. It might be called by the Both Encoder.
    