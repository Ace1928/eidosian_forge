import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
@classmethod
def _tagged_sent_elt(cls, elt, context):
    return list(filter(lambda x: x is not None, [cls._tagged_word_elt(w, None) for w in xpath(elt, '*', cls.ns)]))