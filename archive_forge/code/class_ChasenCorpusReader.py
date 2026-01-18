import sys
from nltk.corpus.reader import util
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
class ChasenCorpusReader(CorpusReader):

    def __init__(self, root, fileids, encoding='utf8', sent_splitter=None):
        self._sent_splitter = sent_splitter
        CorpusReader.__init__(self, root, fileids, encoding)

    def words(self, fileids=None):
        return concat([ChasenCorpusView(fileid, enc, False, False, False, self._sent_splitter) for fileid, enc in self.abspaths(fileids, True)])

    def tagged_words(self, fileids=None):
        return concat([ChasenCorpusView(fileid, enc, True, False, False, self._sent_splitter) for fileid, enc in self.abspaths(fileids, True)])

    def sents(self, fileids=None):
        return concat([ChasenCorpusView(fileid, enc, False, True, False, self._sent_splitter) for fileid, enc in self.abspaths(fileids, True)])

    def tagged_sents(self, fileids=None):
        return concat([ChasenCorpusView(fileid, enc, True, True, False, self._sent_splitter) for fileid, enc in self.abspaths(fileids, True)])

    def paras(self, fileids=None):
        return concat([ChasenCorpusView(fileid, enc, False, True, True, self._sent_splitter) for fileid, enc in self.abspaths(fileids, True)])

    def tagged_paras(self, fileids=None):
        return concat([ChasenCorpusView(fileid, enc, True, True, True, self._sent_splitter) for fileid, enc in self.abspaths(fileids, True)])