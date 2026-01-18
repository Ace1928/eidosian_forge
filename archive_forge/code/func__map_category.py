import functools
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, concat
def _map_category(self, cat):
    pos = cat.find('>')
    if pos == -1:
        return cat
    else:
        return cat[pos + 1:]