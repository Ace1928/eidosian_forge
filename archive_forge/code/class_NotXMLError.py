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
class NotXMLError(ValueError):
    """Failed to parse file as XML."""

    def __init__(self, message):
        """Initialize the class."""
        self.msg = message

    def __str__(self):
        """Return a string summary of the exception."""
        return 'Failed to parse the XML data (%s). Please make sure that the input data are in XML format.' % self.msg