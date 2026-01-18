import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
class MixinProtocol(_sentence._PositioningSentenceProducerMixin):
    """
    A simple, fake protocol that declaratively tells you the sentences
    it produces using L{base.PositioningSentenceProducerMixin}.
    """
    _SENTENCE_CONTENTS = {None: [sentinelValueOne, sentinelValueTwo, None]}