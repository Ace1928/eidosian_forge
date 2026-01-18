import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
class SoapHeader(WsdlMessageReference, SoapParameter):
    """Class for soap:header bindings."""
    part = None

    def __init__(self, elem, wsdl_document):
        super(SoapHeader, self).__init__(elem, wsdl_document)
        if self.message is not None and 'part' in elem.attrib:
            try:
                self.part = self.message.parts[elem.attrib['part']]
            except KeyError:
                msg = 'missing message part {!r}'
                wsdl_document.parse_error(msg.format(elem.attrib['part']))
        if elem.tag == SOAP_HEADER:
            self.faults = [SoapHeaderFault(e, wsdl_document) for e in elem.iterfind(SOAP_HEADERFAULT)]