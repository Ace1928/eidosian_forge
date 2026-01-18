import collections
from ._caveat import error_caveat
from ._utils import condition_with_prefix
class Namespace:
    """Holds maps from schema URIs to prefixes.

    prefixes that are used to encode them in first party
    caveats. Several different URIs may map to the same
    prefix - this is usual when several different backwardly
    compatible schema versions are registered.
    """

    def __init__(self, uri_to_prefix=None):
        self._uri_to_prefix = {}
        if uri_to_prefix is not None:
            for k in uri_to_prefix:
                self.register(k, uri_to_prefix[k])

    def __str__(self):
        """Returns the namespace representation as returned by serialize
        :return: str
        """
        return self.serialize_text().decode('utf-8')

    def __eq__(self, other):
        return self._uri_to_prefix == other._uri_to_prefix

    def serialize_text(self):
        """Returns a serialized form of the Namepace.

        All the elements in the namespace are sorted by
        URI, joined to the associated prefix with a colon and
        separated with spaces.
        :return: bytes
        """
        if self._uri_to_prefix is None or len(self._uri_to_prefix) == 0:
            return b''
        od = collections.OrderedDict(sorted(self._uri_to_prefix.items()))
        data = []
        for uri in od:
            data.append(uri + ':' + od[uri])
        return ' '.join(data).encode('utf-8')

    def register(self, uri, prefix):
        """Registers the given URI and associates it with the given prefix.

        If the URI has already been registered, this is a no-op.

        :param uri: string
        :param prefix: string
        """
        if not is_valid_schema_uri(uri):
            raise KeyError('cannot register invalid URI {} (prefix {})'.format(uri, prefix))
        if not is_valid_prefix(prefix):
            raise ValueError('cannot register invalid prefix %q for URI %q'.format(prefix, uri))
        if self._uri_to_prefix.get(uri) is None:
            self._uri_to_prefix[uri] = prefix

    def resolve(self, uri):
        """ Returns the prefix associated to the uri.

        returns None if not found.
        :param uri: string
        :return: string
        """
        return self._uri_to_prefix.get(uri)

    def resolve_caveat(self, cav):
        """ Resolves the given caveat(string) by using resolve to map from its
        schema namespace to the appropriate prefix.
        If there is no registered prefix for the namespace, it returns an error
        caveat.
        If cav.namespace is empty or cav.location is non-empty, it returns cav
        unchanged.

        It does not mutate ns and may be called concurrently with other
        non-mutating Namespace methods.
        :return: Caveat object
        """
        if cav.namespace == '' or cav.location != '':
            return cav
        prefix = self.resolve(cav.namespace)
        if prefix is None:
            err_cav = error_caveat('caveat {} in unregistered namespace {}'.format(cav.condition, cav.namespace))
            if err_cav.namespace != cav.namespace:
                prefix = self.resolve(err_cav.namespace)
                if prefix is None:
                    prefix = ''
            cav = err_cav
        if prefix != '':
            cav.condition = condition_with_prefix(prefix, cav.condition)
        cav.namespace = ''
        return cav