import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _detab(self, text):
    """Iterate text line by line and convert tabs to spaces.

            >>> m = Markdown()
            >>> m._detab("\\tfoo")
            '    foo'
            >>> m._detab("  \\tfoo")
            '    foo'
            >>> m._detab("\\t  foo")
            '      foo'
            >>> m._detab("  foo")
            '  foo'
            >>> m._detab("  foo\\n\\tbar\\tblam")
            '  foo\\n    bar blam'
        """
    if '\t' not in text:
        return text
    output = []
    for line in text.splitlines():
        output.append(self._detab_line(line))
    return '\n'.join(output)