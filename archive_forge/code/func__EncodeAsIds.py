from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def _EncodeAsIds(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece):
    return _sentencepiece.SentencePieceProcessor__EncodeAsIds(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)