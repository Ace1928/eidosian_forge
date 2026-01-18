from zope.interface import Interface
class IMessageIMAPPart(Interface):

    def getHeaders(negate, *names):
        """
        Retrieve a group of message headers.

        @type names: L{tuple} of L{str}
        @param names: The names of the headers to retrieve or omit.

        @type negate: L{bool}
        @param negate: If True, indicates that the headers listed in C{names}
            should be omitted from the return value, rather than included.

        @rtype: L{dict}
        @return: A mapping of header field names to header field values
        """

    def getBodyFile():
        """
        Retrieve a file object containing only the body of this message.
        """

    def getSize():
        """
        Retrieve the total size, in octets, of this message.

        @rtype: L{int}
        """

    def isMultipart():
        """
        Indicate whether this message has subparts.

        @rtype: L{bool}
        """

    def getSubPart(part):
        """
        Retrieve a MIME sub-message

        @type part: L{int}
        @param part: The number of the part to retrieve, indexed from 0.

        @raise IndexError: Raised if the specified part does not exist.
        @raise TypeError: Raised if this message is not multipart.

        @rtype: Any object implementing L{IMessageIMAPPart}.
        @return: The specified sub-part.
        """