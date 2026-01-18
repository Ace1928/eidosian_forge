import os
from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import add, and_
from nltk.data import show_cfg
from nltk.inference.mace import MaceCommand
from nltk.inference.prover9 import Prover9Command
from nltk.parse import load_parser
from nltk.parse.malt import MaltParser
from nltk.sem.drt import AnaphoraResolutionException, resolve_anaphora
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Expression
from nltk.tag import RegexpTagger
def _construct_readings(self):
    """
        Use ``self._sentences`` to construct a value for ``self._readings``.
        """
    self._readings = {}
    for sid in sorted(self._sentences):
        sentence = self._sentences[sid]
        readings = self._get_readings(sentence)
        self._readings[sid] = {f'{sid}-r{rid}': reading.simplify() for rid, reading in enumerate(sorted(readings, key=str))}