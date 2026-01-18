import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def abs_line_offset(self):
    """Return line offset of current line, from beginning of file."""
    return self.line_offset + self.input_offset