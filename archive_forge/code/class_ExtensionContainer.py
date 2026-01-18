import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
class ExtensionContainer:
    c_tag = ''
    c_namespace = ''

    def __init__(self, text=None, extension_elements=None, extension_attributes=None):
        self.text = text
        self.extension_elements = extension_elements or []
        self.extension_attributes = extension_attributes or {}
        self.encrypted_assertion = None

    def harvest_element_tree(self, tree):
        for child in tree:
            self._convert_element_tree_to_member(child)
        for attribute, value in iter(tree.attrib.items()):
            self._convert_element_attribute_to_member(attribute, value)
        self.text = tree.text

    def _convert_element_tree_to_member(self, child_tree):
        self.extension_elements.append(_extension_element_from_element_tree(child_tree))

    def _convert_element_attribute_to_member(self, attribute, value):
        self.extension_attributes[attribute] = value

    def _add_members_to_element_tree(self, tree):
        for child in self.extension_elements:
            child.become_child_element_of(tree)
        for attribute, value in iter(self.extension_attributes.items()):
            tree.attrib[attribute] = value
        tree.text = self.text

    def find_extensions(self, tag=None, namespace=None):
        """Searches extension elements for child nodes with the desired name.

        Returns a list of extension elements within this object whose tag
        and/or namespace match those passed in. To find all extensions in
        a particular namespace, specify the namespace but not the tag name.
        If you specify only the tag, the result list may contain extension
        elements in multiple namespaces.

        :param tag: str (optional) The desired tag
        :param namespace: str (optional) The desired namespace

        :Return: A list of elements whose tag and/or namespace match the
            parameters values
        """
        results = []
        if tag and namespace:
            for element in self.extension_elements:
                if element.tag == tag and element.namespace == namespace:
                    results.append(element)
        elif tag and (not namespace):
            for element in self.extension_elements:
                if element.tag == tag:
                    results.append(element)
        elif namespace and (not tag):
            for element in self.extension_elements:
                if element.namespace == namespace:
                    results.append(element)
        else:
            for element in self.extension_elements:
                results.append(element)
        return results

    def extensions_as_elements(self, tag, schema):
        """Return extensions that has the given tag and belongs to the
        given schema as native elements of that schema.

        :param tag: The tag of the element
        :param schema: Which schema the element should originate from
        :return: a list of native elements
        """
        result = []
        for ext in self.find_extensions(tag, schema.NAMESPACE):
            ets = schema.ELEMENT_FROM_STRING[tag]
            result.append(ets(ext.to_string()))
        return result

    def add_extension_elements(self, items):
        for item in items:
            self.extension_elements.append(element_to_extension_element(item))

    def add_extension_element(self, item):
        self.extension_elements.append(element_to_extension_element(item))

    def add_extension_attribute(self, name, value):
        self.extension_attributes[name] = value