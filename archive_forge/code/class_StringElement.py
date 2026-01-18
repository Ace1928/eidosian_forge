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
class StringElement(str):
    """NCBI Entrez XML element mapped to a string."""

    def __new__(cls, value, *args, **kwargs):
        """Create a StringElement."""
        return str.__new__(cls, value)

    def __init__(self, value, tag, attributes, key):
        """Initialize a StringElement."""
        self.tag = tag
        self.attributes = attributes
        self.key = key

    def __repr__(self):
        """Return a string representation of the object."""
        text = str.__repr__(self)
        attributes = self.attributes
        if not attributes:
            return text
        return f'StringElement({text}, attributes={attributes!r})'