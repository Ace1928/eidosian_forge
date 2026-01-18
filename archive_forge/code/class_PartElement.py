from suds import *
from suds.sax import Namespace
from suds.sax.document import Document
from suds.sax.element import Element
from suds.sudsobject import Factory
from suds.mx import Content
from suds.mx.literal import Literal as MxLiteral
from suds.umx.typed import Typed as UmxTyped
from suds.bindings.multiref import MultiRef
from suds.xsd.query import TypeQuery, ElementQuery
from suds.xsd.sxbasic import Element as SchemaElement
from suds.options import Options
from suds.plugin import PluginContainer
from copy import deepcopy
class PartElement(SchemaElement):
    """
    Message part referencing an XSD type and thus acting like an XSD element.

    @ivar resolved: The part type.
    @type resolved: L{suds.xsd.sxbase.SchemaObject}

    """

    def __init__(self, name, resolved):
        """
        @param name: The part name.
        @type name: str
        @param resolved: The part type.
        @type resolved: L{suds.xsd.sxbase.SchemaObject}

        """
        root = Element('element', ns=Namespace.xsdns)
        SchemaElement.__init__(self, resolved.schema, root)
        self.__resolved = resolved
        self.name = name
        self.form_qualified = False

    def implany(self):
        pass

    def optional(self):
        return True

    def namespace(self, prefix=None):
        return Namespace.default

    def resolve(self, nobuiltin=False):
        if nobuiltin and self.__resolved.builtin():
            return self
        return self.__resolved