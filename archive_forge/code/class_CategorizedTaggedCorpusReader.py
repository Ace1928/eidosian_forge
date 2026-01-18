import os
from nltk.corpus.reader.api import *
from nltk.corpus.reader.timit import read_timit_block
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
from nltk.tokenize import *
class CategorizedTaggedCorpusReader(CategorizedCorpusReader, TaggedCorpusReader):
    """
    A reader for part-of-speech tagged corpora whose documents are
    divided into categories based on their file identifiers.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining arguments
        are passed to the ``TaggedCorpusReader``.
        """
        CategorizedCorpusReader.__init__(self, kwargs)
        TaggedCorpusReader.__init__(self, *args, **kwargs)

    def tagged_words(self, fileids=None, categories=None, tagset=None):
        return super().tagged_words(self._resolve(fileids, categories), tagset)

    def tagged_sents(self, fileids=None, categories=None, tagset=None):
        return super().tagged_sents(self._resolve(fileids, categories), tagset)

    def tagged_paras(self, fileids=None, categories=None, tagset=None):
        return super().tagged_paras(self._resolve(fileids, categories), tagset)