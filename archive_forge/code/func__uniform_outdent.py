import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
@staticmethod
def _uniform_outdent(text, min_outdent=None, max_outdent=None):
    """
        Removes the smallest common leading indentation from each (non empty)
        line of `text` and returns said indent along with the outdented text.

        Args:
            min_outdent: make sure the smallest common whitespace is at least this size
            max_outdent: the maximum amount a line can be outdented by
        """
    whitespace = [re.findall('^[ \\t]*', line)[0] if line else None for line in text.splitlines()]
    whitespace_not_empty = [i for i in whitespace if i is not None]
    if not whitespace_not_empty:
        return ('', text)
    outdent = min(whitespace_not_empty)
    if min_outdent is not None:
        outdent = min([i for i in whitespace_not_empty if i >= min_outdent] or [min_outdent])
    if max_outdent is not None:
        outdent = min(outdent, max_outdent)
    outdented = []
    for line_ws, line in zip(whitespace, text.splitlines(True)):
        if line.startswith(outdent):
            outdented.append(line.replace(outdent, '', 1))
        elif line_ws is not None and line_ws < outdent:
            outdented.append(line.replace(line_ws, '', 1))
        else:
            outdented.append(line)
    return (outdent, ''.join(outdented))