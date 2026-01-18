import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def _convert_element_tree_to_member(self, child_tree):
    if child_tree.tag in self.__class__.c_children:
        member_name = self.__class__.c_children[child_tree.tag][0]
        member_class = self.__class__.c_children[child_tree.tag][1]
        if isinstance(member_class, list):
            if getattr(self, member_name) is None:
                setattr(self, member_name, [])
            getattr(self, member_name).append(create_class_from_element_tree(member_class[0], child_tree))
        else:
            setattr(self, member_name, create_class_from_element_tree(member_class, child_tree))
    else:
        ExtensionContainer._convert_element_tree_to_member(self, child_tree)