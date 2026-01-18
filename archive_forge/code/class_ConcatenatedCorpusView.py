import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree
from nltk.data import (
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence
class ConcatenatedCorpusView(AbstractLazySequence):
    """
    A 'view' of a corpus file that joins together one or more
    ``StreamBackedCorpusViews<StreamBackedCorpusView>``.  At most
    one file handle is left open at any time.
    """

    def __init__(self, corpus_views):
        self._pieces = corpus_views
        'A list of the corpus subviews that make up this\n        concatenation.'
        self._offsets = [0]
        'A list of offsets, indicating the index at which each\n        subview begins.  In particular::\n            offsets[i] = sum([len(p) for p in pieces[:i]])'
        self._open_piece = None
        'The most recently accessed corpus subview (or None).\n        Before a new subview is accessed, this subview will be closed.'

    def __len__(self):
        if len(self._offsets) <= len(self._pieces):
            for tok in self.iterate_from(self._offsets[-1]):
                pass
        return self._offsets[-1]

    def close(self):
        for piece in self._pieces:
            piece.close()

    def iterate_from(self, start_tok):
        piecenum = bisect.bisect_right(self._offsets, start_tok) - 1
        while piecenum < len(self._pieces):
            offset = self._offsets[piecenum]
            piece = self._pieces[piecenum]
            if self._open_piece is not piece:
                if self._open_piece is not None:
                    self._open_piece.close()
                self._open_piece = piece
            yield from piece.iterate_from(max(0, start_tok - offset))
            if piecenum + 1 == len(self._offsets):
                self._offsets.append(self._offsets[-1] + len(piece))
            piecenum += 1