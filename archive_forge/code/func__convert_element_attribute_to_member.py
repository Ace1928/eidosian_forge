import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def _convert_element_attribute_to_member(self, attribute, value):
    if attribute in self.__class__.c_attributes:
        setattr(self, self.__class__.c_attributes[attribute][0], value)
    else:
        ExtensionContainer._convert_element_attribute_to_member(self, attribute, value)