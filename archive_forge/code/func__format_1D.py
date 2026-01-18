import os
import string
import numpy as np
from Bio.File import as_handle
def _format_1D(self, fmt):
    _alphabet = self._alphabet
    n = len(_alphabet)
    words = [None] * n
    lines = []
    try:
        header = self.header
    except AttributeError:
        pass
    else:
        for line in header:
            line = '#  %s\n' % line
            lines.append(line)
    maxwidth = 0
    for i, key in enumerate(_alphabet):
        value = self[key]
        word = fmt % value
        width = len(word)
        if width > maxwidth:
            maxwidth = width
        words[i] = word
    fmt2 = ' %' + str(maxwidth) + 's'
    for letter, word in zip(_alphabet, words):
        word = fmt2 % word
        line = letter + word + '\n'
        lines.append(line)
    text = ''.join(lines)
    return text