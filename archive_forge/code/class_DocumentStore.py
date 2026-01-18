import suds
class DocumentStore(object):
    """
    Local XML document content repository.

    Each XML document is identified by its location, i.e. URL without any
    protocol identifier. Contained XML documents can be looked up using any URL
    referencing that same location.

    """

    def __init__(self, *args, **kwargs):
        self.__store = {'schemas.xmlsoap.org/soap/encoding/': soap5_encoding_schema}
        self.update = self.__store.update
        self.update(*args, **kwargs)

    def __len__(self):
        return len(self.__store)

    def open(self, url):
        """
        Open a document at the specified URL.

        The document URL's needs not contain a protocol identifier, and if it
        does, that protocol identifier is ignored when looking up the store
        content.

        Missing documents referenced using the internal 'suds' protocol are
        reported by raising an exception. For other protocols, None is returned
        instead.

        @param url: A document URL.
        @type url: str
        @return: Document content or None if not found.
        @rtype: bytes

        """
        protocol, location = self.__split(url)
        content = self.__find(location)
        if protocol == 'suds' and content is None:
            raise Exception('location "%s" not in document store' % location)
        return content

    def __find(self, location):
        """
        Find the specified location in the store.

        @param location: The I{location} part of a URL.
        @type location: str
        @return: Document content or None if not found.
        @rtype: bytes

        """
        return self.__store.get(location)

    def __split(self, url):
        """
        Split the given URL into its I{protocol} & I{location} components.

        @param url: A URL.
        @param url: str
        @return: (I{protocol}, I{location})
        @rtype: (str, str)

        """
        parts = url.split('://', 1)
        if len(parts) == 2:
            return parts
        return (None, url)