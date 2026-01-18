from ..exceptions import UndeclaredNamespace
class _StrictFeedParser(object):

    def __init__(self, baseuri, baselang, encoding):
        self.bozo = 0
        self.exc = None
        self.decls = {}
        self.baseuri = baseuri or ''
        self.lang = baselang
        self.encoding = encoding
        super(_StrictFeedParser, self).__init__()

    @staticmethod
    def _normalize_attributes(kv):
        k = kv[0].lower()
        v = k in ('rel', 'type') and kv[1].lower() or kv[1]
        return (k, v)

    def startPrefixMapping(self, prefix, uri):
        if not uri:
            return
        prefix = prefix or None
        self.track_namespace(prefix, uri)
        if prefix and uri == 'http://www.w3.org/1999/xlink':
            self.decls['xmlns:' + prefix] = uri

    def startElementNS(self, name, qname, attrs):
        namespace, localname = name
        lowernamespace = str(namespace or '').lower()
        if lowernamespace.find('backend.userland.com/rss') != -1:
            namespace = 'http://backend.userland.com/rss'
            lowernamespace = namespace
        if qname and qname.find(':') > 0:
            givenprefix = qname.split(':')[0]
        else:
            givenprefix = None
        prefix = self._matchnamespaces.get(lowernamespace, givenprefix)
        if givenprefix and (prefix is None or (prefix == '' and lowernamespace == '')) and (givenprefix not in self.namespaces_in_use):
            raise UndeclaredNamespace("'%s' is not associated with a namespace" % givenprefix)
        localname = str(localname).lower()
        attrsD, self.decls = (self.decls, {})
        if localname == 'math' and namespace == 'http://www.w3.org/1998/Math/MathML':
            attrsD['xmlns'] = namespace
        if localname == 'svg' and namespace == 'http://www.w3.org/2000/svg':
            attrsD['xmlns'] = namespace
        if prefix:
            localname = prefix.lower() + ':' + localname
        elif namespace and (not qname):
            for name, value in self.namespaces_in_use.items():
                if name and value == namespace:
                    localname = name + ':' + localname
                    break
        for (namespace, attrlocalname), attrvalue in attrs.items():
            lowernamespace = (namespace or '').lower()
            prefix = self._matchnamespaces.get(lowernamespace, '')
            if prefix:
                attrlocalname = prefix + ':' + attrlocalname
            attrsD[str(attrlocalname).lower()] = attrvalue
        for qname in attrs.getQNames():
            attrsD[str(qname).lower()] = attrs.getValueByQName(qname)
        localname = str(localname).lower()
        self.unknown_starttag(localname, list(attrsD.items()))

    def characters(self, text):
        self.handle_data(text)

    def endElementNS(self, name, qname):
        namespace, localname = name
        lowernamespace = str(namespace or '').lower()
        if qname and qname.find(':') > 0:
            givenprefix = qname.split(':')[0]
        else:
            givenprefix = ''
        prefix = self._matchnamespaces.get(lowernamespace, givenprefix)
        if prefix:
            localname = prefix + ':' + localname
        elif namespace and (not qname):
            for name, value in self.namespaces_in_use.items():
                if name and value == namespace:
                    localname = name + ':' + localname
                    break
        localname = str(localname).lower()
        self.unknown_endtag(localname)

    def error(self, exc):
        self.bozo = 1
        self.exc = exc
    warning = error

    def fatalError(self, exc):
        self.error(exc)
        raise exc