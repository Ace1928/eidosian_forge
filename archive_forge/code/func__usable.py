from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
def _usable(self, encoding, tried):
    """Should we even bother to try this encoding?

        :param encoding: Name of an encoding.
        :param tried: Encodings that have already been tried. This will be modified
            as a side effect.
        """
    if encoding is not None:
        encoding = encoding.lower()
        if encoding in self.exclude_encodings:
            return False
        if encoding not in tried:
            tried.add(encoding)
            return True
    return False