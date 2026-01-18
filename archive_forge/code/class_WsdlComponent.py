import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class WsdlComponent:

    def __init__(self, elem, wsdl_document):
        self.elem = elem
        self.wsdl_document = wsdl_document
        self.name = get_qname(wsdl_document.target_namespace, elem.get('name'))

    def __repr__(self):
        return '%s(name=%r)' % (self.__class__.__name__, self.prefixed_name)

    def get(self, name):
        return self.elem.get(name)

    @property
    def attrib(self):
        return self.elem.attrib

    @property
    def local_name(self):
        if self.name:
            return local_name(self.name)

    @property
    def prefixed_name(self):
        if self.name:
            return get_prefixed_qname(self.name, self.wsdl_document.namespaces)

    def map_qname(self, qname):
        return get_prefixed_qname(qname, self.wsdl_document.namespaces)

    def unmap_qname(self, qname):
        return get_extended_qname(qname, self.wsdl_document.namespaces)

    def _parse_reference(self, elem, attribute_name):
        try:
            return self.unmap_qname(elem.attrib[attribute_name])
        except KeyError:
            return