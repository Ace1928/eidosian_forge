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
class OriginInfo(collections.namedtuple('OriginInfo', ('loc', 'function_name', 'source_code_line', 'comment'))):
    """Container for information about the source code before conversion.

  Attributes:
    loc: Location
    function_name: Optional[Text]
    source_code_line: Text
    comment: Optional[Text]
  """

    def as_frame(self):
        """Returns a 4-tuple consistent with the return of traceback.extract_tb."""
        return (self.loc.filename, self.loc.lineno, self.function_name, self.source_code_line)

    def __repr__(self):
        if self.loc.filename:
            return '{}:{}:{}'.format(os.path.split(self.loc.filename)[1], self.loc.lineno, self.loc.col_offset)
        return '<no file>:{}:{}'.format(self.loc.lineno, self.loc.col_offset)