from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import (
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.translate import AlignedSent, Alignment

        :return: the given file(s) as a list of AlignedSent objects.
        :rtype: list(AlignedSent)
        