import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
class SamlBase(ExtensionContainer):
    """A foundation class on which SAML classes are built. It
    handles the parsing of attributes and children which are common to all
    SAML classes. By default, the SamlBase class translates all XML child
    nodes into ExtensionElements.
    """
    c_children: Any = {}
    c_attributes: Any = {}
    c_attribute_type: Any = {}
    c_child_order: list[str] = []
    c_cardinality: dict[str, dict[str, int]] = {}
    c_any: Optional[dict[str, str]] = None
    c_any_attribute: Optional[dict[str, str]] = None
    c_value_type: Any = None
    c_ns_prefix = None

    def _get_all_c_children_with_order(self):
        if len(self.c_child_order) > 0:
            yield from self.c_child_order
        else:
            for _, values in iter(self.__class__.c_children.items()):
                yield values[0]

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

    def _convert_element_attribute_to_member(self, attribute, value):
        if attribute in self.__class__.c_attributes:
            setattr(self, self.__class__.c_attributes[attribute][0], value)
        else:
            ExtensionContainer._convert_element_attribute_to_member(self, attribute, value)

    def _add_members_to_element_tree(self, tree):
        for member_name in self._get_all_c_children_with_order():
            member = getattr(self, member_name)
            if member is None:
                pass
            elif isinstance(member, list):
                for instance in member:
                    instance.become_child_element_of(tree)
            else:
                member.become_child_element_of(tree)
        for xml_attribute, attribute_info in iter(self.__class__.c_attributes.items()):
            member_name, member_type, required = attribute_info
            member = getattr(self, member_name)
            if member is not None:
                tree.attrib[xml_attribute] = member
        ExtensionContainer._add_members_to_element_tree(self, tree)

    def become_child_element_of(self, node):
        """
        Note: Only for use with classes that have a c_tag and c_namespace class
        member. It is in SamlBase so that it can be inherited but it should
        not be called on instances of SamlBase.

        :param node: The node to which this instance should be a child
        """
        new_child = self._to_element_tree()
        node.append(new_child)

    def _to_element_tree(self):
        """

        Note, this method is designed to be used only with classes that have a
        c_tag and c_namespace. It is placed in SamlBase for inheritance but
        should not be called on in this class.

        """
        new_tree = ElementTree.Element(f'{{{self.__class__.c_namespace}}}{self.__class__.c_tag}')
        self._add_members_to_element_tree(new_tree)
        return new_tree

    def register_prefix(self, nspair):
        """
        Register with ElementTree a set of namespaces

        :param nspair: A dictionary of prefixes and uris to use when
            constructing the text representation.
        :return:
        """
        for prefix, uri in nspair.items():
            try:
                ElementTree.register_namespace(prefix, uri)
            except AttributeError:
                ElementTree._namespace_map[uri] = prefix
            except ValueError:
                pass

    def get_ns_map_attribute(self, attributes, uri_set):
        for attribute in attributes:
            if attribute[0] == '{':
                uri, tag = attribute[1:].split('}')
                uri_set.add(uri)
        return uri_set

    def tag_get_uri(self, elem):
        if elem.tag[0] == '{':
            uri, tag = elem.tag[1:].split('}')
            return uri
        return None

    def get_ns_map(self, elements, uri_set):
        for elem in elements:
            uri_set = self.get_ns_map_attribute(elem.attrib, uri_set)
            children = list(elem)
            uri_set = self.get_ns_map(children, uri_set)
            uri = self.tag_get_uri(elem)
            if uri is not None:
                uri_set.add(uri)
        return uri_set

    def get_prefix_map(self, elements):
        uri_set = self.get_ns_map(elements, set())
        prefix_map = {}
        for uri in sorted(uri_set):
            prefix_map[f'encas{len(prefix_map)}'] = uri
        return prefix_map

    def get_xml_string_with_self_contained_assertion_within_advice_encrypted_assertion(self, assertion_tag, advice_tag):
        for tmp_encrypted_assertion in self.assertion.advice.encrypted_assertion:
            if tmp_encrypted_assertion.encrypted_data is None:
                prefix_map = self.get_prefix_map([tmp_encrypted_assertion._to_element_tree().find(assertion_tag)])
                tree = self._to_element_tree()
                encs = tree.find(assertion_tag).find(advice_tag).findall(tmp_encrypted_assertion._to_element_tree().tag)
                for enc in encs:
                    assertion = enc.find(assertion_tag)
                    if assertion is not None:
                        self.set_prefixes(assertion, prefix_map)
        return ElementTree.tostring(tree, encoding='UTF-8').decode('utf-8')

    def get_xml_string_with_self_contained_assertion_within_encrypted_assertion(self, assertion_tag):
        """Makes a encrypted assertion only containing self contained
        namespaces.

        :param assertion_tag: Tag for the assertion to be transformed.
        :return: A new samlp.Resonse in string representation.
        """
        prefix_map = self.get_prefix_map([self.encrypted_assertion._to_element_tree().find(assertion_tag)])
        tree = self._to_element_tree()
        self.set_prefixes(tree.find(self.encrypted_assertion._to_element_tree().tag).find(assertion_tag), prefix_map)
        return ElementTree.tostring(tree, encoding='UTF-8').decode('utf-8')

    def set_prefixes(self, elem, prefix_map):
        if not ElementTree.iselement(elem):
            elem = elem.getroot()
        uri_map = {}
        for prefix, uri in prefix_map.items():
            uri_map[uri] = prefix
            elem.set(f'xmlns:{prefix}', uri)
        memo = {}
        for element in elem.iter():
            self.fixup_element_prefixes(element, uri_map, memo)

    def fixup_element_prefixes(self, elem, uri_map, memo):

        def fixup(name):
            try:
                return memo[name]
            except KeyError:
                if name[0] != '{':
                    return
                uri, tag = name[1:].split('}')
                if uri in uri_map:
                    new_name = f'{uri_map[uri]}:{tag}'
                    memo[name] = new_name
                    return new_name
        name = fixup(elem.tag)
        if name:
            elem.tag = name
        for key, value in elem.items():
            name = fixup(key)
            if name:
                elem.set(name, value)
                del elem.attrib[key]

    def to_string_force_namespace(self, nspair):
        elem = self._to_element_tree()
        self.set_prefixes(elem, nspair)
        return ElementTree.tostring(elem, encoding='UTF-8')

    def to_string(self, nspair=None):
        """Converts the Saml object to a string containing XML.

        :param nspair: A dictionary of prefixes and uris to use when
            constructing the text representation.
        :return: String representation of the object
        """
        if not nspair and self.c_ns_prefix:
            nspair = self.c_ns_prefix
        if nspair:
            self.register_prefix(nspair)
        return ElementTree.tostring(self._to_element_tree(), encoding='UTF-8')

    def __str__(self):
        x = self.to_string()
        if not isinstance(x, str):
            x = x.decode('utf-8')
        return x

    def keyswv(self):
        """Return the keys of attributes or children that has values

        :return: list of keys
        """
        return [key for key, val in self.__dict__.items() if val]

    def keys(self):
        """Return all the keys that represent possible attributes and
        children.

        :return: list of keys
        """
        keys = ['text']
        keys.extend([n for n, t, r in self.c_attributes.values()])
        keys.extend([v[0] for v in self.c_children.values()])
        return keys

    def children_with_values(self):
        """Returns all children that has values

        :return: Possibly empty list of children.
        """
        childs = []
        for attribute in self._get_all_c_children_with_order():
            member = getattr(self, attribute)
            if member is None or member == []:
                pass
            elif isinstance(member, list):
                for instance in member:
                    childs.append(instance)
            else:
                childs.append(member)
        return childs

    def set_text(self, val, base64encode=False):
        """Sets the text property of this instance.

        :param val: The value of the text property
        :param base64encode: Whether the value should be base64encoded
        :return: The instance
        """
        if isinstance(val, bool):
            self.text = 'true' if val else 'false'
        elif isinstance(val, int):
            self.text = str(val)
        elif isinstance(val, str):
            self.text = val
        elif val is None:
            pass
        else:
            raise ValueError(f"Type shouldn't be '{val}'")
        return self

    def loadd(self, ava, base64encode=False):
        """
        Sets attributes, children, extension elements and extension
        attributes of this element instance depending on what is in
        the given dictionary. If there are already values on properties
        those will be overwritten. If the keys in the dictionary does
        not correspond to known attributes/children/.. they are ignored.

        :param ava: The dictionary
        :param base64encode: Whether the values on attributes or texts on
            children shoule be base64encoded.
        :return: The instance
        """
        for prop, _typ, _req in self.c_attributes.values():
            if prop in ava:
                value = ava[prop]
                if isinstance(value, (bool, int)):
                    setattr(self, prop, str(value))
                else:
                    setattr(self, prop, value)
        if 'text' in ava:
            self.set_text(ava['text'], base64encode)
        for prop, klassdef in self.c_children.values():
            if prop in ava:
                if isinstance(klassdef, list):
                    make_vals(ava[prop], klassdef[0], self, prop, base64encode=base64encode)
                else:
                    cis = make_vals(ava[prop], klassdef, self, prop, True, base64encode)
                    setattr(self, prop, cis)
        if 'extension_elements' in ava:
            for item in ava['extension_elements']:
                self.extension_elements.append(ExtensionElement(item['tag']).loadd(item))
        if 'extension_attributes' in ava:
            for key, val in ava['extension_attributes'].items():
                self.extension_attributes[key] = val
        return self

    def clear_text(self):
        if self.text:
            _text = self.text.strip()
            if _text == '':
                self.text = None

    def __eq__(self, other):
        if not isinstance(other, SamlBase):
            return False
        self.clear_text()
        other.clear_text()
        if len(self.keyswv()) != len(other.keyswv()):
            return False
        for key in self.keyswv():
            if key in ['_extatt']:
                continue
            svals = self.__dict__[key]
            ovals = other.__dict__[key]
            if isinstance(svals, str):
                if svals != ovals:
                    return False
            elif isinstance(svals, list):
                for sval in svals:
                    try:
                        for oval in ovals:
                            if sval == oval:
                                break
                        else:
                            return False
                    except TypeError:
                        return False
            elif svals == ovals:
                pass
            else:
                return False
        return True

    def child_class(self, child):
        """Return the class a child element should be an instance of

        :param child: The name of the child element
        :return: The class
        """
        for prop, klassdef in self.c_children.values():
            if child == prop:
                if isinstance(klassdef, list):
                    return klassdef[0]
                else:
                    return klassdef
        return None

    def child_cardinality(self, child):
        """Return the cardinality of a child element

        :param child: The name of the child element
        :return: The cardinality as a 2-tuple (min, max).
            The max value is either a number or the string "unbounded".
            The min value is always a number.
        """
        for prop, klassdef in self.c_children.values():
            if child == prop:
                if isinstance(klassdef, list):
                    try:
                        _min = self.c_cardinality['min']
                    except KeyError:
                        _min = 1
                    try:
                        _max = self.c_cardinality['max']
                    except KeyError:
                        _max = 'unbounded'
                    return (_min, _max)
                else:
                    return (1, 1)
        return None

    def verify(self):
        return valid_instance(self)

    def empty(self):
        for prop, _typ, _req in self.c_attributes.values():
            if getattr(self, prop, None):
                return False
        for prop, klassdef in self.c_children.values():
            if getattr(self, prop):
                return False
        for param in ['text', 'extension_elements', 'extension_attributes']:
            if getattr(self, param):
                return False
        return True