import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
def _parse_services(self):
    for child in self.iterfind(WSDL_SERVICE):
        service = WsdlService(child, self)
        if service.name in self.maps.services:
            self.parse_error('duplicated service {!r}'.format(service.prefixed_name))
        else:
            self.maps.services[service.name] = service