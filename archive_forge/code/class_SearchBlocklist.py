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
class SearchBlocklist(object):
    """
    Search block list facilitates blocking ngrams from being generated.
    """

    def __init__(self, dict_agent: DictionaryAgent) -> None:
        self.dict = dict_agent
        self._phrases: Set[str] = set()
        self._phrase_ngrams: Dict[int, List[List[int]]] = {}

    def __bool__(self):
        return bool(self._phrases)

    def clear(self) -> None:
        self._phrases = set()
        self._phrase_ngrams = {}

    def _add_literal(self, phrase_literal: str):
        if phrase_literal in self._phrases:
            return
        ngram = self.dict.txt2vec(phrase_literal)
        self._phrases.add(phrase_literal)
        logging.debug(f"Adding '{phrase_literal}' to the beam block_list {ngram}")
        l = len(ngram)
        if l not in self._phrase_ngrams:
            self._phrase_ngrams[l] = []
        self._phrase_ngrams[l].append(ngram)

    def add(self, phrase: str):
        phrase = phrase.strip()
        if not phrase:
            return
        self._add_literal(phrase)
        self._add_literal(phrase + 's')
        self._add_literal(phrase.lower())
        self._add_literal(phrase.lower() + 's')
        self._add_literal(phrase.upper())
        self._add_literal(phrase.upper() + 'S')
        self._add_literal(phrase.title())
        self._add_literal(phrase.title() + 'S')
        self._add_literal(phrase[0].upper() + phrase[1:])
        self._add_literal(phrase[0].upper() + phrase[1:] + 's')
        self._add_literal(phrase[0].upper() + phrase[1:].lower())
        self._add_literal(phrase[0].upper() + phrase[1:].lower() + 's')

    def items(self) -> Iterable[Tuple[int, List[List[int]]]]:
        return self._phrase_ngrams.items()