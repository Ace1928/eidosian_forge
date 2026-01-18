import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _strip_footnote_definitions(self, text):
    """A footnote definition looks like this:

            [^note-id]: Text of the note.

                May include one or more indented paragraphs.

        Where,
        - The 'note-id' can be pretty much anything, though typically it
          is the number of the footnote.
        - The first paragraph may start on the next line, like so:

            [^note-id]:
                Text of the note.
        """
    less_than_tab = self.tab_width - 1
    footnote_def_re = re.compile('\n            ^[ ]{0,%d}\\[\\^(.+)\\]:   # id = \\1\n            [ \\t]*\n            (                       # footnote text = \\2\n              # First line need not start with the spaces.\n              (?:\\s*.*\\n+)\n              (?:\n                (?:[ ]{%d} | \\t)  # Subsequent lines must be indented.\n                .*\\n+\n              )*\n            )\n            # Lookahead for non-space at line-start, or end of doc.\n            (?:(?=^[ ]{0,%d}\\S)|\\Z)\n            ' % (less_than_tab, self.tab_width, self.tab_width), re.X | re.M)
    return footnote_def_re.sub(self._extract_footnote_def_sub, text)