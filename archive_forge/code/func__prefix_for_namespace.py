from io import BytesIO
from io import StringIO
from lxml import etree
from bs4.element import (
from bs4.builder import (
from bs4.dammit import EncodingDetector
def _prefix_for_namespace(self, namespace):
    """Find the currently active prefix for the given namespace."""
    if namespace is None:
        return None
    for inverted_nsmap in reversed(self.nsmaps):
        if inverted_nsmap is not None and namespace in inverted_nsmap:
            return inverted_nsmap[namespace]
    return None