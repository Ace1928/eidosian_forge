class QualifiedName(Identifier):
    """Qualified name of an identifier in a particular namespace."""

    def __init__(self, namespace, localpart):
        """
        Constructor.

        :param namespace: Namespace to use for qualified name resolution.
        :param localpart: Portion of identifier not part of the namespace prefix.
        """
        Identifier.__init__(self, ''.join([namespace.uri, localpart]))
        self._namespace = namespace
        self._localpart = localpart
        self._str = ':'.join([namespace.prefix, localpart]) if namespace.prefix else localpart

    @property
    def namespace(self):
        """Namespace of qualified name."""
        return self._namespace

    @property
    def localpart(self):
        """Local part of qualified name."""
        return self._localpart

    def __str__(self):
        return self._str

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self._str)

    def __hash__(self):
        return hash(self.uri)

    def provn_representation(self):
        """PROV-N representation of qualified name in a string."""
        return "'%s'" % self._str