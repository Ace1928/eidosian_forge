import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
class MixinTests(TestCase, SentenceTestsMixin):
    """
    Tests for protocols deriving from L{base.PositioningSentenceProducerMixin}
    and their sentences.
    """

    def setUp(self):
        self.protocol = MixinProtocol()
        self.sentenceClass = MixinSentence

    def test_noNoneInSentenceAttributes(self):
        """
        L{None} does not appear in the sentence attributes of the
        protocol, even though it's in the specification.

        This is because L{None} is a placeholder for parts of the sentence you
        don't really need or want, but there are some bits later on in the
        sentence that you do want. The alternative would be to have to specify
        things like "_UNUSED0", "_UNUSED1"... which would end up cluttering
        the sentence data and eventually adapter state.
        """
        sentenceAttributes = self.protocol.getSentenceAttributes()
        self.assertNotIn(None, sentenceAttributes)
        sentenceContents = self.protocol._SENTENCE_CONTENTS
        sentenceSpecAttributes = itertools.chain(*sentenceContents.values())
        self.assertIn(None, sentenceSpecAttributes)