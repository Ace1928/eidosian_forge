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
class DictionaryElement(dict):
    """NCBI Entrez XML element mapped to a dictionaray."""

    def __init__(self, tag, attrs, allowed_tags, repeated_tags=None, key=None):
        """Create a DictionaryElement."""
        self.tag = tag
        if key is None:
            self.key = tag
        else:
            self.key = key
        self.attributes = attrs
        self.allowed_tags = allowed_tags
        self.repeated_tags = repeated_tags
        if repeated_tags:
            for key in repeated_tags:
                self[key] = []

    def __repr__(self):
        """Return a string representation of the object."""
        text = dict.__repr__(self)
        attributes = self.attributes
        if not attributes:
            return text
        return f'DictElement({text}, attributes={attributes!r})'

    def store(self, value):
        """Add an entry to the dictionary, checking tags."""
        key = value.key
        tag = value.tag
        if self.allowed_tags is not None and tag not in self.allowed_tags:
            raise ValueError("Unexpected item '%s' in dictionary" % key)
        del value.key
        if self.repeated_tags and key in self.repeated_tags:
            self[key].append(value)
        else:
            self[key] = value