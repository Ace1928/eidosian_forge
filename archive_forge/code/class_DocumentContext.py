from suds import *
from logging import getLogger
class DocumentContext(Context):
    """
    The XML document load context.

    @ivar url: The URL.
    @type url: str
    @ivar document: Either the XML text or the B{parsed} document root.
    @type document: (str|L{sax.element.Element})

    """
    pass