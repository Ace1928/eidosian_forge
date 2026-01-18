import suds.cache
import suds.plugin
import suds.sax.parser
import suds.transport
class DocumentReader(Reader):
    """Integrates between the SAX L{Parser} and the document cache."""

    def open(self, url):
        """
        Open an XML document at the specified I{URL}.

        First, a preparsed document is looked up in the I{object cache}. If not
        found, its content is fetched from an external source and parsed using
        the SAX parser. The result is cached for the next open().

        @param url: A document URL.
        @type url: str.
        @return: The specified XML document.
        @rtype: I{Document}

        """
        cache = self.__cache()
        id = self.mangle(url, 'document')
        xml = cache.get(id)
        if xml is None:
            xml = self.__fetch(url)
            cache.put(id, xml)
        self.plugins.document.parsed(url=url, document=xml.root())
        return xml

    def __cache(self):
        """
        Get the I{object cache}.

        @return: The I{cache} when I{cachingpolicy} = B{0}.
        @rtype: L{Cache}

        """
        if self.options.cachingpolicy == 0:
            return self.options.cache
        return suds.cache.NoCache()

    def __fetch(self, url):
        """
        Fetch document content from an external source.

        The document content will first be looked up in the registered document
        store, and if not found there, downloaded using the registered
        transport system.

        Before being returned, the fetched document content first gets
        processed by all the registered 'loaded' plugins.

        @param url: A document URL.
        @type url: str.
        @return: A file pointer to the fetched document content.
        @rtype: file-like

        """
        content = None
        store = self.options.documentStore
        if store is not None:
            content = store.open(url)
        if content is None:
            request = suds.transport.Request(url)
            request.headers = self.options.headers
            fp = self.options.transport.open(request)
            try:
                content = fp.read()
            finally:
                fp.close()
        ctx = self.plugins.document.loaded(url=url, document=content)
        content = ctx.document
        sax = suds.sax.parser.Parser()
        return sax.parse(string=content)