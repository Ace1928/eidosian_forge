import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def _to_element_tree(self):
    """

        Note, this method is designed to be used only with classes that have a
        c_tag and c_namespace. It is placed in SamlBase for inheritance but
        should not be called on in this class.

        """
    new_tree = ElementTree.Element(f'{{{self.__class__.c_namespace}}}{self.__class__.c_tag}')
    self._add_members_to_element_tree(new_tree)
    return new_tree