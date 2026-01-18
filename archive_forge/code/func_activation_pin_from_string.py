import saml2
from saml2 import SamlBase
def activation_pin_from_string(xml_string):
    return saml2.create_class_from_xml_string(ActivationPin, xml_string)