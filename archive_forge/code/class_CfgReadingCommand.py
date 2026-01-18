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
class CfgReadingCommand(ReadingCommand):

    def __init__(self, gramfile=None):
        """
        :param gramfile: name of file where grammar can be loaded
        :type gramfile: str
        """
        self._gramfile = gramfile if gramfile else 'grammars/book_grammars/discourse.fcfg'
        self._parser = load_parser(self._gramfile)

    def parse_to_readings(self, sentence):
        """:see: ReadingCommand.parse_to_readings()"""
        from nltk.sem import root_semrep
        tokens = sentence.split()
        trees = self._parser.parse(tokens)
        return [root_semrep(tree) for tree in trees]

    def combine_readings(self, readings):
        """:see: ReadingCommand.combine_readings()"""
        return reduce(and_, readings)

    def to_fol(self, expression):
        """:see: ReadingCommand.to_fol()"""
        return expression