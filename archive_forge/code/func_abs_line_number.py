import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def abs_line_number(self):
    """Return line number of current line (counting from 1)."""
    return self.line_offset + self.input_offset + 1