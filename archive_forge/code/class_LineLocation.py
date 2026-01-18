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
class LineLocation(collections.namedtuple('LineLocation', ('filename', 'lineno'))):
    """Similar to Location, but without column information.

  Attributes:
    filename: Text
    lineno: int, 1-based
  """
    pass