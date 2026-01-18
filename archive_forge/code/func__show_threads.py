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
def _show_threads(self, filter=False, show_thread_readings=False):
    """
        Print out the value of ``self._threads`` or ``self._filtered_hreads``
        """
    threads = self._filtered_threads if filter else self._threads
    for tid in sorted(threads):
        if show_thread_readings:
            readings = [self._readings[rid.split('-')[0]][rid] for rid in self._threads[tid]]
            try:
                thread_reading = ': %s' % self._reading_command.combine_readings(readings).normalize()
            except Exception as e:
                thread_reading = ': INVALID: %s' % e.__class__.__name__
        else:
            thread_reading = ''
        print('%s:' % tid, self._threads[tid], thread_reading)