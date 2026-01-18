from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.util import Index
class CMUDictCorpusReader(CorpusReader):

    def entries(self):
        """
        :return: the cmudict lexicon as a list of entries
            containing (word, transcriptions) tuples.
        """
        return concat([StreamBackedCorpusView(fileid, read_cmudict_block, encoding=enc) for fileid, enc in self.abspaths(None, True)])

    def words(self):
        """
        :return: a list of all words defined in the cmudict lexicon.
        """
        return [word.lower() for word, _ in self.entries()]

    def dict(self):
        """
        :return: the cmudict lexicon as a dictionary, whose keys are
            lowercase words and whose values are lists of pronunciations.
        """
        return dict(Index(self.entries()))