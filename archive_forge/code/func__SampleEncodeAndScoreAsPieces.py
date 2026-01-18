from sys import version_info as _swig_python_version_info
import re
import csv
import sys
import os
from io import StringIO
from io import BytesIO
from ._version import __version__
def _SampleEncodeAndScoreAsPieces(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece):
    return _sentencepiece.SentencePieceProcessor__SampleEncodeAndScoreAsPieces(self, text, num_samples, alpha, wor, include_best, add_bos, add_eos, reverse, emit_unk_piece)