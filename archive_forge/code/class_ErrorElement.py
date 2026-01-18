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
class ErrorElement(str):
    """NCBI Entrez XML element containing an error message."""

    def __new__(cls, value, *args, **kwargs):
        """Create an ErrorElement."""
        return str.__new__(cls, value)

    def __init__(self, value, tag):
        """Initialize an ErrorElement."""
        self.tag = tag
        self.key = tag

    def __repr__(self):
        """Return the error message as a string."""
        text = str.__repr__(self)
        return f'ErrorElement({text})'