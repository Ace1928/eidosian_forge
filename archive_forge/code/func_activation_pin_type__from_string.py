import saml2
from saml2 import SamlBase
def activation_pin_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(ActivationPinType_, xml_string)