import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
def _tagged_words_block_reader(self, stream, tagset=None):
    return sum(self._tagged_discourses_block_reader(stream, tagset)[0], [])