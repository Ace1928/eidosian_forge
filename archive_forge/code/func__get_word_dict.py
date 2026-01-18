from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def _get_word_dict(lexicon, lm, lm_dict, tokens_dict, unk_word):
    word_dict = None
    if lm_dict is not None:
        word_dict = _Dictionary(lm_dict)
    if lexicon and word_dict is None:
        word_dict = _create_word_dict(lexicon)
    elif not lexicon and word_dict is None and (type(lm) == str):
        d = {tokens_dict.get_entry(i): [[tokens_dict.get_entry(i)]] for i in range(tokens_dict.index_size())}
        d[unk_word] = [[unk_word]]
        word_dict = _create_word_dict(d)
    return word_dict