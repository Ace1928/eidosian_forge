from abc import ABC, abstractmethod
from typing import TypeVar, List, Dict, Optional, Tuple, Set, Iterable
import math
from operator import attrgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.core.torch_agent import TorchAgent, Batch, Output, DictionaryAgent
from parlai.utils.misc import warn_once
import parlai.utils.logging as logging
from parlai.core.metrics import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.utils.torch import (
def _load_beam_block_list(self) -> SearchBlocklist:
    """
        Load the beam block_list.

        :return: a dict mapping ngram length to different ngrams
        """
    block_list = SearchBlocklist(self.dict)
    if not self.opt.get('beam_block_list_filename'):
        return block_list
    block_list_fn = self.opt['beam_block_list_filename']
    try:
        with open(block_list_fn) as f:
            for line in f:
                block_list.add(line.strip())
    except IOError:
        logging.error(f'Could not load beam block_list {block_list_fn}, using empty block_list.')
    return block_list