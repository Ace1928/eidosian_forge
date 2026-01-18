from importlib import import_module
import logging
import os
import sys
from saml2 import NAMESPACE
from saml2 import ExtensionElement
from saml2 import SAMLError
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2.s_utils import do_ava
from saml2.s_utils import factory
from saml2.saml import NAME_FORMAT_UNSPECIFIED
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def _create_nameid_ext_el(value):
    text = value['text'] if isinstance(value, dict) else value
    attributes = {'Format': NAMEID_FORMAT_PERSISTENT, 'NameQualifier': value['NameQualifier'], 'SPNameQualifier': value['SPNameQualifier']} if isinstance(value, dict) else {'Format': NAMEID_FORMAT_PERSISTENT}
    element = ExtensionElement('NameID', NAMESPACE, attributes=attributes, text=text)
    return element