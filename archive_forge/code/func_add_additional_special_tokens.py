from parlai.core.opt import Opt
from parlai.core.build_data import modelzoo_path
from parlai.utils.bpe import bpe_factory, BPEHelper
from .agents import Agent
from .build_data import make_dir
from collections import defaultdict
import codecs
import copy
import numpy as np
import os
import json
import re
import parlai.utils.logging as logging
from typing import List
def add_additional_special_tokens(self, additional_special_tokens: List[str]):
    """
        Add additional special tokens to the dictionary.

        Should only be called after initialization of the existing dictionary.
        """
    self.additional_special_tokens = additional_special_tokens
    if self.additional_special_tokens and (not self.supports_additional_special_tokens()):
        raise RuntimeError(f'{self.tokenizer} does not currently support adding additional special tokens')
    for tok in self.additional_special_tokens:
        self.add_token(tok)
    for i, tok in enumerate(self.additional_special_tokens):
        self.freq[tok] = 1000000000 + 4 + i
    if self.tokenizer == 'bytelevelbpe':
        self.bpe.add_special_tokens(self, self.additional_special_tokens)