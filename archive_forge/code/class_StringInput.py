import sys
import os
import re
import codecs
from docutils import TransformSpec
from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
class StringInput(Input):
    """
    Direct string input.
    """
    default_source_path = '<string>'

    def read(self):
        """Decode and return the source string."""
        return self.decode(self.source)