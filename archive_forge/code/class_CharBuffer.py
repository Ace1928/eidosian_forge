from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
class CharBuffer:
    """Helps reading NEXUS-words and characters from a buffer (semi-PRIVATE).

    This class is not intended for public use (any more).
    """

    def __init__(self, string):
        """Initialize the class."""
        if string:
            self.buffer = list(string)
        else:
            self.buffer = []

    def peek(self):
        """Return the first character from the buffer."""
        if self.buffer:
            return self.buffer[0]
        else:
            return None

    def peek_nonwhitespace(self):
        """Return the first character from the buffer, do not include spaces."""
        b = ''.join(self.buffer).strip()
        if b:
            return b[0]
        else:
            return None

    def __next__(self):
        """Iterate over NEXUS characters in the file."""
        if self.buffer:
            return self.buffer.pop(0)
        else:
            return None

    def next_nonwhitespace(self):
        """Check for next non whitespace character in NEXUS file."""
        while True:
            p = next(self)
            if p is None:
                break
            if p not in WHITESPACE:
                return p
        return None

    def skip_whitespace(self):
        """Skip whitespace characters in NEXUS file."""
        while self.buffer[0] in WHITESPACE:
            self.buffer = self.buffer[1:]

    def next_until(self, target):
        """Iterate over the NEXUS file until a target character is reached."""
        for t in target:
            try:
                pos = self.buffer.index(t)
            except ValueError:
                pass
            else:
                found = ''.join(self.buffer[:pos])
                self.buffer = self.buffer[pos:]
                return found
        else:
            return None

    def peek_word(self, word):
        """Return a word stored in the buffer."""
        return ''.join(self.buffer[:len(word)]) == word

    def next_word(self):
        """Return the next NEXUS word from a string.

        This deals with single and double quotes, whitespace and punctuation.
        """
        word = []
        quoted = False
        first = self.next_nonwhitespace()
        if not first:
            return None
        word.append(first)
        if first == "'":
            quoted = "'"
        elif first == '"':
            quoted = '"'
        elif first in PUNCTUATION:
            return first
        while True:
            c = self.peek()
            if c == quoted:
                word.append(next(self))
                if self.peek() == quoted:
                    next(self)
                elif quoted:
                    break
            elif quoted:
                word.append(next(self))
            elif not c or c in PUNCTUATION or c in WHITESPACE:
                break
            else:
                word.append(next(self))
        return ''.join(word)

    def rest(self):
        """Return the rest of the string without parsing."""
        return ''.join(self.buffer)