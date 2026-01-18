import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
class DummySentence(_sentence._BaseSentence):
    """
    A sentence for L{DummyProtocol}.
    """
    ALLOWED_ATTRIBUTES = DummyProtocol.getSentenceAttributes()