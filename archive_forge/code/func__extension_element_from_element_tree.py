import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def _extension_element_from_element_tree(element_tree):
    elementc_tag = element_tree.tag
    if '}' in elementc_tag:
        namespace = elementc_tag[1:elementc_tag.index('}')]
        tag = elementc_tag[elementc_tag.index('}') + 1:]
    else:
        namespace = None
        tag = elementc_tag
    extension = ExtensionElement(namespace=namespace, tag=tag)
    for key, value in iter(element_tree.attrib.items()):
        extension.attributes[key] = value
    for child in element_tree:
        extension.children.append(_extension_element_from_element_tree(child))
    extension.text = element_tree.text
    return extension