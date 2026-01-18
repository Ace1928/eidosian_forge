import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
from ...models.bert.tokenization_bert import whitespace_tokenize
from ...tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
from ...utils import is_tf_available, is_torch_available, logging
from .utils import DataProcessor
class SquadV1Processor(SquadProcessor):
    train_file = 'train-v1.1.json'
    dev_file = 'dev-v1.1.json'