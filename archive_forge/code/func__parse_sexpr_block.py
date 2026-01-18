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
def _parse_sexpr_block(block):
    tokens = []
    start = end = 0
    while end < len(block):
        m = re.compile('\\S').search(block, end)
        if not m:
            return (tokens, end)
        start = m.start()
        if m.group() != '(':
            m2 = re.compile('[\\s(]').search(block, start)
            if m2:
                end = m2.start()
            else:
                if tokens:
                    return (tokens, end)
                raise ValueError('Block too small')
        else:
            nesting = 0
            for m in re.compile('[()]').finditer(block, start):
                if m.group() == '(':
                    nesting += 1
                else:
                    nesting -= 1
                if nesting == 0:
                    end = m.end()
                    break
            else:
                if tokens:
                    return (tokens, end)
                raise ValueError('Block too small')
        tokens.append(block[start:end])
    return (tokens, end)