from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
class Locator:
    """Interface for associating a SAX event with a document
    location. A locator object will return valid results only during
    calls to DocumentHandler methods; at any other time, the
    results are unpredictable."""

    def getColumnNumber(self):
        """Return the column number where the current event ends."""
        return -1

    def getLineNumber(self):
        """Return the line number where the current event ends."""
        return -1

    def getPublicId(self):
        """Return the public identifier for the current event."""
        return None

    def getSystemId(self):
        """Return the system identifier for the current event."""
        return None