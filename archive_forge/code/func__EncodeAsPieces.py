from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def _EncodeAsPieces(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
    return _sentencepiece.SentencePieceProcessor__EncodeAsPieces(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)