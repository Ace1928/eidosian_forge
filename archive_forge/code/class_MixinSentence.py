import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
class MixinSentence(_sentence._BaseSentence):
    """
    A sentence for L{MixinProtocol}.
    """
    ALLOWED_ATTRIBUTES = MixinProtocol.getSentenceAttributes()