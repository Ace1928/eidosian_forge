from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
class XMLReader:
    """Interface for reading an XML document using callbacks.

    XMLReader is the interface that an XML parser's SAX2 driver must
    implement. This interface allows an application to set and query
    features and properties in the parser, to register event handlers
    for document processing, and to initiate a document parse.

    All SAX interfaces are assumed to be synchronous: the parse
    methods must not return until parsing is complete, and readers
    must wait for an event-handler callback to return before reporting
    the next event."""

    def __init__(self):
        self._cont_handler = handler.ContentHandler()
        self._dtd_handler = handler.DTDHandler()
        self._ent_handler = handler.EntityResolver()
        self._err_handler = handler.ErrorHandler()

    def parse(self, source):
        """Parse an XML document from a system identifier or an InputSource."""
        raise NotImplementedError('This method must be implemented!')

    def getContentHandler(self):
        """Returns the current ContentHandler."""
        return self._cont_handler

    def setContentHandler(self, handler):
        """Registers a new object to receive document content events."""
        self._cont_handler = handler

    def getDTDHandler(self):
        """Returns the current DTD handler."""
        return self._dtd_handler

    def setDTDHandler(self, handler):
        """Register an object to receive basic DTD-related events."""
        self._dtd_handler = handler

    def getEntityResolver(self):
        """Returns the current EntityResolver."""
        return self._ent_handler

    def setEntityResolver(self, resolver):
        """Register an object to resolve external entities."""
        self._ent_handler = resolver

    def getErrorHandler(self):
        """Returns the current ErrorHandler."""
        return self._err_handler

    def setErrorHandler(self, handler):
        """Register an object to receive error-message events."""
        self._err_handler = handler

    def setLocale(self, locale):
        """Allow an application to set the locale for errors and warnings.

        SAX parsers are not required to provide localization for errors
        and warnings; if they cannot support the requested locale,
        however, they must raise a SAX exception. Applications may
        request a locale change in the middle of a parse."""
        raise SAXNotSupportedException('Locale support not implemented')

    def getFeature(self, name):
        """Looks up and returns the state of a SAX2 feature."""
        raise SAXNotRecognizedException("Feature '%s' not recognized" % name)

    def setFeature(self, name, state):
        """Sets the state of a SAX2 feature."""
        raise SAXNotRecognizedException("Feature '%s' not recognized" % name)

    def getProperty(self, name):
        """Looks up and returns the value of a SAX2 property."""
        raise SAXNotRecognizedException("Property '%s' not recognized" % name)

    def setProperty(self, name, value):
        """Sets the value of a SAX2 property."""
        raise SAXNotRecognizedException("Property '%s' not recognized" % name)