import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
def _parse_bindings(self):
    for child in self.iterfind(WSDL_BINDING):
        binding = WsdlBinding(child, self)
        if binding.name in self.maps.bindings:
            self.parse_error('duplicated binding {!r}'.format(binding.prefixed_name))
        else:
            self.maps.bindings[binding.name] = binding