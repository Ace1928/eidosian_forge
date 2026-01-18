import sys
import os
import re
import codecs
from docutils import TransformSpec
from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
class BinaryFileOutput(FileOutput):
    """
    A version of docutils.io.FileOutput which writes to a binary file.
    """
    mode = 'wb'