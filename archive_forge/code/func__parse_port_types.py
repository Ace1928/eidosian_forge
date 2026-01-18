import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
def _parse_port_types(self):
    for child in self.iterfind(WSDL_PORT_TYPE):
        port_type = WsdlPortType(child, self)
        if port_type.name in self.maps.port_types:
            self.parse_error('duplicated port type {!r}'.format(port_type.prefixed_name))
        else:
            self.maps.port_types[port_type.name] = port_type