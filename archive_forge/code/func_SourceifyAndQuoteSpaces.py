import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def SourceifyAndQuoteSpaces(path):
    """Convert a path to its source directory form and quote spaces."""
    return QuoteSpaces(Sourceify(path))