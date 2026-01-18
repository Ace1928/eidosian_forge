import collections
import difflib
import io
import os
import tokenize
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.util import tf_inspect
def _absolute_col_offset(self, col_offset):
    if col_offset is None:
        return 0
    return col_offset + self._col_offset