from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import line_tokenize
class SwadeshCorpusReader(WordListCorpusReader):

    def entries(self, fileids=None):
        """
        :return: a tuple of words for the specified fileids.
        """
        if not fileids:
            fileids = self.fileids()
        wordlists = [self.words(f) for f in fileids]
        return list(zip(*wordlists))