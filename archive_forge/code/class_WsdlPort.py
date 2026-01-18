import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class WsdlPort(WsdlComponent):
    binding = None
    soap_location = None

    def __init__(self, elem, wsdl_document):
        super(WsdlPort, self).__init__(elem, wsdl_document)
        binding_name = self._parse_reference(elem, 'binding')
        try:
            self.binding = wsdl_document.maps.bindings[binding_name]
        except KeyError:
            if binding_name:
                msg = 'unknown binding {!r} for {!r} output'
                wsdl_document.parse_error(msg.format(binding_name, self))
        if wsdl_document.soap_binding:
            for child in elem.iterfind(SOAP_ADDRESS):
                self.soap_location = child.get('location')
                break