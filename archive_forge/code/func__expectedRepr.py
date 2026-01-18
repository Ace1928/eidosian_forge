import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
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