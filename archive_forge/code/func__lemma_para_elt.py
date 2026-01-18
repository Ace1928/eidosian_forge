import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
@classmethod
def _lemma_para_elt(cls, elt, context):
    return [cls._lemma_sent_elt(s, None) for s in xpath(elt, '*', cls.ns)]