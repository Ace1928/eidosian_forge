import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class Wsdl11Maps:

    def __init__(self, wsdl_document):
        self.wsdl_document = wsdl_document
        self.imports = {}
        self.messages = {}
        self.port_types = {}
        self.bindings = {}
        self.services = {}

    def clear(self):
        self.imports.clear()
        self.messages.clear()
        self.port_types.clear()
        self.bindings.clear()
        self.services.clear()