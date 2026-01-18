from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader, XMLCorpusView
class BNCSentence(list):
    """
    A list of words, augmented by an attribute ``num`` used to record
    the sentence identifier (the ``n`` attribute from the XML).
    """

    def __init__(self, num, items):
        self.num = num
        list.__init__(self, items)