from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def _construct_trie(tokens_dict, word_dict, lexicon, lm, silence):
    vocab_size = tokens_dict.index_size()
    trie = _Trie(vocab_size, silence)
    start_state = lm.start(False)
    for word, spellings in lexicon.items():
        word_idx = word_dict.get_index(word)
        _, score = lm.score(start_state, word_idx)
        for spelling in spellings:
            spelling_idx = [tokens_dict.get_index(token) for token in spelling]
            trie.insert(spelling_idx, word_idx, score)
    trie.smear(_SmearingMode.MAX)
    return trie