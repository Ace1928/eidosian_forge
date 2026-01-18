class EntityResolver:
    """Basic interface for resolving entities. If you create an object
    implementing this interface, then register the object with your
    Parser, the parser will call the method in your object to
    resolve all external entities. Note that DefaultHandler implements
    this interface with the default behaviour."""

    def resolveEntity(self, publicId, systemId):
        """Resolve the system identifier of an entity and return either
        the system identifier to read from as a string, or an InputSource
        to read from."""
        return systemId