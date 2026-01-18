import sys
import os
import re
import warnings
import types
import unicodedata
class Targetable(Resolvable):
    referenced = 0
    indirect_reference_name = None
    'Holds the whitespace_normalized_name (contains mixed case) of a target.\n    Required for MoinMoin/reST compatibility.'