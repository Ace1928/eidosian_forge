import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class WsdlPortType(WsdlComponent):

    def __init__(self, elem, wsdl_document):
        super(WsdlPortType, self).__init__(elem, wsdl_document)
        self.operations = {}
        for child in elem.iterfind(WSDL_OPERATION):
            operation_name = child.get('name')
            if operation_name is None:
                continue
            operation = WsdlOperation(child, wsdl_document)
            key = operation.key
            if key in self.operations:
                msg = 'duplicated operation {!r} for {!r}'
                wsdl_document.parse_error(msg.format(operation_name, self))
            self.operations[key] = operation