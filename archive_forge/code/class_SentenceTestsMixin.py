import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
class SentenceTestsMixin:
    """
    Tests for positioning protocols and their respective sentences.
    """

    def test_attributeAccess(self):
        """
        A sentence attribute gets the correct value, and accessing an
        unset attribute (which is specified as being a valid sentence
        attribute) gets L{None}.
        """
        thisSentinel = object()
        sentence = self.sentenceClass({sentinelValueOne: thisSentinel})
        self.assertEqual(getattr(sentence, sentinelValueOne), thisSentinel)
        self.assertIsNone(getattr(sentence, sentinelValueTwo))

    def test_raiseOnMissingAttributeAccess(self):
        """
        Accessing a nonexistent attribute raises C{AttributeError}.
        """
        sentence = self.sentenceClass({})
        self.assertRaises(AttributeError, getattr, sentence, 'BOGUS')

    def test_raiseOnBadAttributeAccess(self):
        """
        Accessing bogus attributes raises C{AttributeError}, *even*
        when that attribute actually is in the sentence data.
        """
        sentence = self.sentenceClass({'BOGUS': None})
        self.assertRaises(AttributeError, getattr, sentence, 'BOGUS')
    sentenceType = 'tummies'
    reprTemplate = '<%s (%s) {%s}>'

    def _expectedRepr(self, sentenceType='unknown type', dataRepr=''):
        """
        Builds the expected repr for a sentence.

        @param sentenceType: The name of the sentence type (e.g "GPGGA").
        @type sentenceType: C{str}
        @param dataRepr: The repr of the data in the sentence.
        @type dataRepr: C{str}
        @return: The expected repr of the sentence.
        @rtype: C{str}
        """
        clsName = self.sentenceClass.__name__
        return self.reprTemplate % (clsName, sentenceType, dataRepr)

    def test_unknownTypeRepr(self):
        """
        Test the repr of an empty sentence of unknown type.
        """
        sentence = self.sentenceClass({})
        expectedRepr = self._expectedRepr()
        self.assertEqual(repr(sentence), expectedRepr)

    def test_knownTypeRepr(self):
        """
        Test the repr of an empty sentence of known type.
        """
        sentence = self.sentenceClass({'type': self.sentenceType})
        expectedRepr = self._expectedRepr(self.sentenceType)
        self.assertEqual(repr(sentence), expectedRepr)