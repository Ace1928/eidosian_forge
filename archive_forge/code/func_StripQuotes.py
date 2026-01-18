from __future__ import print_function
import os
import re
import sys
def StripQuotes(s):
    s, stripped = StripDelimiters(s, '"', '"')
    if not stripped:
        s, stripped = StripDelimiters(s, "'", "'")
    return s