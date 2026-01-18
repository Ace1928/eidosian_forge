import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
def _read_word_block(self, stream):
    words = []
    for sent in self._read_sent_block(stream):
        words.extend(sent)
    return words