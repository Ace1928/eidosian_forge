from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
class PdfFormatError(RuntimeError):
    """An error that probably indicates a syntactic or semantic error in the
    PDF file structure"""
    pass