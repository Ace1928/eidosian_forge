import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
def _read_keyword_block(self, stream):
    keywords = []
    for comparison in self._read_comparison_block(stream):
        keywords.append(comparison.keyword)
    return keywords