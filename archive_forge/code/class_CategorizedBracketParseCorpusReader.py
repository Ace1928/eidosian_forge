import sys
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
class CategorizedBracketParseCorpusReader(CategorizedCorpusReader, BracketParseCorpusReader):
    """
    A reader for parsed corpora whose documents are
    divided into categories based on their file identifiers.
    @author: Nathan Schneider <nschneid@cs.cmu.edu>
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (C{cat_pattern}, C{cat_map}, and C{cat_file}) are passed to
        the L{CategorizedCorpusReader constructor
        <CategorizedCorpusReader.__init__>}.  The remaining arguments
        are passed to the L{BracketParseCorpusReader constructor
        <BracketParseCorpusReader.__init__>}.
        """
        CategorizedCorpusReader.__init__(self, kwargs)
        BracketParseCorpusReader.__init__(self, *args, **kwargs)

    def tagged_words(self, fileids=None, categories=None, tagset=None):
        return super().tagged_words(self._resolve(fileids, categories), tagset)

    def tagged_sents(self, fileids=None, categories=None, tagset=None):
        return super().tagged_sents(self._resolve(fileids, categories), tagset)

    def tagged_paras(self, fileids=None, categories=None, tagset=None):
        return super().tagged_paras(self._resolve(fileids, categories), tagset)

    def parsed_words(self, fileids=None, categories=None):
        return super().parsed_words(self._resolve(fileids, categories))

    def parsed_sents(self, fileids=None, categories=None):
        return super().parsed_sents(self._resolve(fileids, categories))

    def parsed_paras(self, fileids=None, categories=None):
        return super().parsed_paras(self._resolve(fileids, categories))