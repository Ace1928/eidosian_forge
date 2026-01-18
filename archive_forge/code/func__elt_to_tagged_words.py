import re
import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *
from nltk.internals import ElementWrapper
from nltk.tag import map_tag
from nltk.util import LazyConcatenation
def _elt_to_tagged_words(self, elt, handler, tagset=None):
    tagged_post = [(self._simplify_username(t.attrib['word']), t.attrib['pos']) for t in elt.findall('t')]
    if tagset and tagset != self._tagset:
        tagged_post = [(w, map_tag(self._tagset, tagset, t)) for w, t in tagged_post]
    return tagged_post