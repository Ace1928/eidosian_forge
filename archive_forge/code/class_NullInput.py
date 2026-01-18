import sys
import os
import re
import codecs
from docutils import TransformSpec
from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
class NullInput(Input):
    """
    Degenerate input: read nothing.
    """
    default_source_path = 'null input'

    def read(self):
        """Return a null string."""
        return ''