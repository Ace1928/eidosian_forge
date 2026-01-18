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
class DrtGlueReadingCommand(ReadingCommand):

    def __init__(self, semtype_file=None, remove_duplicates=False, depparser=None):
        """
        :param semtype_file: name of file where grammar can be loaded
        :param remove_duplicates: should duplicates be removed?
        :param depparser: the dependency parser
        """
        if semtype_file is None:
            semtype_file = os.path.join('grammars', 'sample_grammars', 'drt_glue.semtype')
        self._glue = DrtGlue(semtype_file=semtype_file, remove_duplicates=remove_duplicates, depparser=depparser)

    def parse_to_readings(self, sentence):
        """:see: ReadingCommand.parse_to_readings()"""
        return self._glue.parse_to_meaning(sentence)

    def process_thread(self, sentence_readings):
        """:see: ReadingCommand.process_thread()"""
        try:
            return [self.combine_readings(sentence_readings)]
        except AnaphoraResolutionException:
            return []

    def combine_readings(self, readings):
        """:see: ReadingCommand.combine_readings()"""
        thread_reading = reduce(add, readings)
        return resolve_anaphora(thread_reading.simplify())

    def to_fol(self, expression):
        """:see: ReadingCommand.to_fol()"""
        return expression.fol()