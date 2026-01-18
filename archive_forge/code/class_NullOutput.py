import sys
import os
import re
import codecs
from docutils import TransformSpec
from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
class NullOutput(Output):
    """
    Degenerate output: write nothing.
    """
    default_destination_path = 'null output'

    def write(self, data):
        """Do nothing ([don't even] send data to the bit bucket)."""
        pass