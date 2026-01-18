from suds import *
from suds.bindings.document import Document
from suds.bindings.rpc import RPC, Encoded
from suds.reader import DocumentReader
from suds.sax.element import Element
from suds.sudsobject import Object, Facade, Metadata
from suds.xsd import qualify, Namespace
from suds.xsd.query import ElementQuery
from suds.xsd.schema import Schema, SchemaCollection
import re
from . import soaparray
from urllib.parse import urljoin
from logging import getLogger
def __resolveport(self, definitions):
    """
        Resolve port_type reference.

        @param definitions: A definitions object.
        @type definitions: L{Definitions}

        """
    ref = qualify(self.type, self.root, definitions.tns)
    port_type = definitions.port_types.get(ref)
    if port_type is None:
        raise Exception("portType '%s', not-found" % (self.type,))
    port_type.resolve(definitions)
    self.type = port_type