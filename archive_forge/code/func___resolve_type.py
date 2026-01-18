from suds import *
from suds.reader import DocumentReader
from suds.sax import Namespace
from suds.transport import TransportError
from suds.xsd import *
from suds.xsd.query import *
from suds.xsd.sxbase import *
from urllib.parse import urljoin
from logging import getLogger
def __resolve_type(self, nobuiltin=False):
    """
        Private resolve() worker without any result caching.

        @param nobuiltin: Flag indicating whether resolving to XSD built-in
            types should not be allowed.
        @return: The resolved (true) type.
        @rtype: L{SchemaObject}

        """
    qref = self.qref()
    if qref is None:
        return self
    query = TypeQuery(qref)
    query.history = [self]
    log.debug('%s, resolving: %s\n using:%s', self.id, qref, query)
    resolved = query.execute(self.schema)
    if resolved is None:
        log.debug(self.schema)
        raise TypeNotFound(qref)
    if resolved.builtin() and nobuiltin:
        return self
    return resolved