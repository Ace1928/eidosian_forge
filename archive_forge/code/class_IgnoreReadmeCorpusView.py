from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus.reader.api import *
class IgnoreReadmeCorpusView(StreamBackedCorpusView):
    """
    This CorpusView is used to skip the initial readme block of the corpus.
    """

    def __init__(self, *args, **kwargs):
        StreamBackedCorpusView.__init__(self, *args, **kwargs)
        self._open()
        read_blankline_block(self._stream)
        self._filepos = [self._stream.tell()]