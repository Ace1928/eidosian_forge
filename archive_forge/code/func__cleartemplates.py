import itertools as it
from abc import ABCMeta, abstractmethod
from nltk.tbl.feature import Feature
from nltk.tbl.rule import Rule
@classmethod
def _cleartemplates(cls):
    cls.ALLTEMPLATES = []