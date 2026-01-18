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
class IntegerElement(int):
    """NCBI Entrez XML element mapped to an integer."""

    def __new__(cls, value, *args, **kwargs):
        """Create an IntegerElement."""
        return int.__new__(cls, value)

    def __init__(self, value, tag, attributes, key):
        """Initialize an IntegerElement."""
        self.tag = tag
        self.attributes = attributes
        self.key = key

    def __repr__(self):
        """Return a string representation of the object."""
        text = int.__repr__(self)
        try:
            attributes = self.attributes
        except AttributeError:
            return text
        return f'IntegerElement({text}, attributes={attributes!r})'