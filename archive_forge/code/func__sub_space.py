import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree
from nltk.data import (
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence
def _sub_space(m):
    """Helper function: given a regexp match, return a string of
    spaces that's the same length as the matched string."""
    return ' ' * (m.end() - m.start())