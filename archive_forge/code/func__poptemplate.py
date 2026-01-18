import itertools as it
from abc import ABCMeta, abstractmethod
from nltk.tbl.feature import Feature
from nltk.tbl.rule import Rule
@classmethod
def _poptemplate(cls):
    return cls.ALLTEMPLATES.pop() if cls.ALLTEMPLATES else None