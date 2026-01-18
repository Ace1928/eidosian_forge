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
class WObject(Object):
    """
    Base object for WSDL types.

    @ivar root: The XML I{root} element.
    @type root: L{Element}

    """

    def __init__(self, root):
        """
        @param root: An XML root element.
        @type root: L{Element}

        """
        Object.__init__(self)
        self.root = root
        pmd = Metadata()
        pmd.excludes = ['root']
        pmd.wrappers = dict(qname=repr)
        self.__metadata__.__print__ = pmd
        self.__resolved = False

    def resolve(self, definitions):
        """
        Resolve named references to other WSDL objects.

        Can be safely called multiple times.

        @param definitions: A definitions object.
        @type definitions: L{Definitions}

        """
        if not self.__resolved:
            self.do_resolve(definitions)
            self.__resolved = True

    def do_resolve(self, definitions):
        """
        Internal worker resolving named references to other WSDL objects.

        May only be called once per instance.

        @param definitions: A definitions object.
        @type definitions: L{Definitions}

        """
        pass