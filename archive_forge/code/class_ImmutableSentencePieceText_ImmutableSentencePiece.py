from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
class ImmutableSentencePieceText_ImmutableSentencePiece(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece_swiginit(self, _sentencepiece.new_ImmutableSentencePieceText_ImmutableSentencePiece())
    __swig_destroy__ = _sentencepiece.delete_ImmutableSentencePieceText_ImmutableSentencePiece

    def _piece(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__piece(self)

    def _surface(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__surface(self)

    def _id(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__id(self)

    def _begin(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__begin(self)

    def _end(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__end(self)

    def _surface_as_bytes(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__surface_as_bytes(self)

    def _piece_as_bytes(self):
        return _sentencepiece.ImmutableSentencePieceText_ImmutableSentencePiece__piece_as_bytes(self)
    piece = property(_piece)
    piece_as_bytes = property(_piece_as_bytes)
    surface = property(_surface)
    surface_as_bytes = property(_surface_as_bytes)
    id = property(_id)
    begin = property(_begin)
    end = property(_end)

    def __str__(self):
        return 'piece: "{}"\nid: {}\nsurface: "{}"\nbegin: {}\nend: {}\n'.format(self.piece, self.id, self.surface, self.begin, self.end)

    def __eq__(self, other):
        return self.piece == other.piece and self.id == other.id and (self.surface == other.surface) and (self.begin == other.begin) and (self.end == other.end)

    def __hash__(self):
        return hash(str(self))
    __repr__ = __str__