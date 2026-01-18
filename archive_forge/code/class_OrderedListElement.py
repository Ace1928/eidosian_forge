import os
import warnings
from collections import Counter
from xml.parsers import expat
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from urllib.request import urlopen
from urllib.parse import urlparse
from Bio import StreamModeError
class OrderedListElement(list):
    """NCBI Entrez XML element mapped to a list of lists.

    OrderedListElement is used to describe a list of repeating elements such as
    A, B, C, A, B, C, A, B, C ... where each set of A, B, C forms a group. This
    is then stored as [[A, B, C], [A, B, C], [A, B, C], ...]
    """

    def __init__(self, tag, attributes, allowed_tags, first_tag, key=None):
        """Create an OrderedListElement."""
        self.tag = tag
        if key is None:
            self.key = tag
        else:
            self.key = key
        self.attributes = attributes
        self.allowed_tags = allowed_tags
        self.first_tag = first_tag

    def __repr__(self):
        """Return a string representation of the object."""
        text = list.__repr__(self)
        attributes = self.attributes
        if not attributes:
            return text
        return f'OrderedListElement({text}, attributes={attributes!r})'

    def store(self, value):
        """Append an element to the list, checking tags."""
        key = value.key
        if self.allowed_tags is not None and key not in self.allowed_tags:
            raise ValueError("Unexpected item '%s' in list" % key)
        if key == self.first_tag:
            self.append([])
        self[-1].append(value)