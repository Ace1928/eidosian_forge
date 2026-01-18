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
def __resolvefaults(self, definitions, op):
    """
        Resolve SOAP fault I{message} references by cross-referencing with
        operations defined in the port type.

        @param definitions: A definitions object.
        @type definitions: L{Definitions}
        @param op: An I{operation} object.
        @type op: I{operation}

        """
    ptop = self.type.operation(op.name)
    if ptop is None:
        raise Exception("operation '%s' not defined in portType" % (op.name,))
    soap = op.soap
    for fault in soap.faults:
        for f in ptop.faults:
            if f.name == fault.name:
                fault.parts = f.message.parts
                continue
        if hasattr(fault, 'parts'):
            continue
        raise Exception("fault '%s' not defined in portType '%s'" % (fault.name, self.type.name))