import suds.cache
import suds.plugin
import suds.sax.parser
import suds.transport
class DefinitionsReader(Reader):
    """
    Integrates between the WSDL Definitions object and the object cache.

    @ivar fn: A factory function used to create objects not found in the cache.
    @type fn: I{Constructor}

    """

    def __init__(self, options, fn):
        """
        @param options: An options object.
        @type options: I{Options}
        @param fn: A factory function used to create objects not found in the
            cache.
        @type fn: I{Constructor}

        """
        super(DefinitionsReader, self).__init__(options)
        self.fn = fn

    def open(self, url):
        """
        Open a WSDL schema at the specified I{URL}.

        First, the WSDL schema is looked up in the I{object cache}. If not
        found, a new one constructed using the I{fn} factory function and the
        result is cached for the next open().

        @param url: A WSDL URL.
        @type url: str.
        @return: The WSDL object.
        @rtype: I{Definitions}

        """
        cache = self.__cache()
        id = self.mangle(url, 'wsdl')
        wsdl = cache.get(id)
        if wsdl is None:
            wsdl = self.fn(url, self.options)
            cache.put(id, wsdl)
        else:
            wsdl.options = self.options
            for imp in wsdl.imports:
                imp.imported.options = self.options
        return wsdl

    def __cache(self):
        """
        Get the I{object cache}.

        @return: The I{cache} when I{cachingpolicy} = B{1}.
        @rtype: L{Cache}

        """
        if self.options.cachingpolicy == 1:
            return self.options.cache
        return suds.cache.NoCache()