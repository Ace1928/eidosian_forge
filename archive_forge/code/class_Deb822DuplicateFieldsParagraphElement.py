import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
class Deb822DuplicateFieldsParagraphElement(Deb822ParagraphElement):

    def __init__(self, kvpair_elements):
        super().__init__()
        self._kvpair_order = LinkedList()
        self._kvpair_elements = {}
        self._init_kvpair_fields(kvpair_elements)
        self._init_parent_of_parts()

    @property
    def has_duplicate_fields(self):
        return len(self._kvpair_order) > len(self._kvpair_elements)

    def _init_kvpair_fields(self, kvpairs):
        assert not self._kvpair_order
        assert not self._kvpair_elements
        for kv in kvpairs:
            field_name = kv.field_name
            node = self._kvpair_order.append(kv)
            if field_name not in self._kvpair_elements:
                self._kvpair_elements[field_name] = [node]
            else:
                self._kvpair_elements[field_name].append(node)

    def _nodes_being_relocated(self, field):
        key, index, name_token = _unpack_key(field)
        nodes = self._kvpair_elements[key]
        nodes_being_relocated = []
        if name_token is not None or index is not None:
            single_node = self._resolve_to_single_node(nodes, key, index, name_token)
            assert single_node is not None
            nodes_being_relocated.append(single_node)
        else:
            nodes_being_relocated = nodes
        return (nodes, nodes_being_relocated)

    def order_last(self, field):
        """Re-order the given field so it is "last" in the paragraph"""
        nodes, nodes_being_relocated = self._nodes_being_relocated(field)
        assert len(nodes_being_relocated) == 1 or len(nodes) == len(nodes_being_relocated)
        kvpair_order = self._kvpair_order
        for node in nodes_being_relocated:
            if kvpair_order.tail_node is node:
                continue
            kvpair_order.remove_node(node)
            assert kvpair_order.tail_node is not None
            kvpair_order.insert_node_after(node, kvpair_order.tail_node)
        if len(nodes_being_relocated) == 1 and nodes_being_relocated[0] is not nodes[-1]:
            single_node = nodes_being_relocated[0]
            nodes.remove(single_node)
            nodes.append(single_node)

    def order_first(self, field):
        """Re-order the given field so it is "first" in the paragraph"""
        nodes, nodes_being_relocated = self._nodes_being_relocated(field)
        assert len(nodes_being_relocated) == 1 or len(nodes) == len(nodes_being_relocated)
        kvpair_order = self._kvpair_order
        for node in nodes_being_relocated:
            if kvpair_order.head_node is node:
                continue
            kvpair_order.remove_node(node)
            assert kvpair_order.head_node is not None
            kvpair_order.insert_node_before(node, kvpair_order.head_node)
        if len(nodes_being_relocated) == 1 and nodes_being_relocated[0] is not nodes[0]:
            single_node = nodes_being_relocated[0]
            nodes.remove(single_node)
            nodes.insert(0, single_node)

    def order_before(self, field, reference_field):
        """Re-order the given field so appears directly after the reference field in the paragraph

        The reference field must be present."""
        nodes, nodes_being_relocated = self._nodes_being_relocated(field)
        assert len(nodes_being_relocated) == 1 or len(nodes) == len(nodes_being_relocated)
        _, reference_nodes = self._nodes_being_relocated(reference_field)
        reference_node = reference_nodes[0]
        if reference_node in nodes_being_relocated:
            raise ValueError('Cannot re-order a field relative to itself')
        kvpair_order = self._kvpair_order
        for node in nodes_being_relocated:
            kvpair_order.remove_node(node)
            kvpair_order.insert_node_before(node, reference_node)
        if len(nodes_being_relocated) == 1 and len(nodes) > 1:
            field_name = nodes_being_relocated[0].value.field_name
            self._regenerate_relative_kvapir_order(field_name)

    def order_after(self, field, reference_field):
        """Re-order the given field so appears directly before the reference field in the paragraph

        The reference field must be present.
        """
        nodes, nodes_being_relocated = self._nodes_being_relocated(field)
        assert len(nodes_being_relocated) == 1 or len(nodes) == len(nodes_being_relocated)
        _, reference_nodes = self._nodes_being_relocated(reference_field)
        reference_node = reference_nodes[-1]
        if reference_node in nodes_being_relocated:
            raise ValueError('Cannot re-order a field relative to itself')
        kvpair_order = self._kvpair_order
        for node in reversed(nodes_being_relocated):
            kvpair_order.remove_node(node)
            kvpair_order.insert_node_after(node, reference_node)
        if len(nodes_being_relocated) == 1 and len(nodes) > 1:
            field_name = nodes_being_relocated[0].value.field_name
            self._regenerate_relative_kvapir_order(field_name)

    def _regenerate_relative_kvapir_order(self, field_name):
        nodes = []
        for node in self._kvpair_order.iter_nodes():
            if node.value.field_name == field_name:
                nodes.append(node)
        self._kvpair_elements[field_name] = nodes

    def iter_parts(self):
        yield from self._kvpair_order

    @property
    def kvpair_count(self):
        return len(self._kvpair_order)

    def iter_keys(self):
        yield from (kv.field_name for kv in self._kvpair_order)

    def _resolve_to_single_node(self, nodes, key, index, name_token, use_get=False):
        if index is None:
            if len(nodes) != 1:
                if name_token is not None:
                    node = self._find_node_via_name_token(name_token, nodes)
                    if node is not None:
                        return node
                msg = 'Ambiguous key {key} - the field appears {res_len} times. Use ({key}, index) to denote which instance of the field you want.  (Index can be 0..{res_len_1} or e.g. -1 to denote the last field)'
                raise AmbiguousDeb822FieldKeyError(msg.format(key=key, res_len=len(nodes), res_len_1=len(nodes) - 1))
            index = 0
        try:
            return nodes[index]
        except IndexError:
            if use_get:
                return None
            msg = 'Field "{key}" was present but the index "{index}" was invalid.'
            raise KeyError(msg.format(key=key, index=index))

    def get_kvpair_element(self, item, use_get=False):
        key, index, name_token = _unpack_key(item)
        if use_get:
            nodes = self._kvpair_elements.get(key)
            if nodes is None:
                return None
        else:
            nodes = self._kvpair_elements[key]
        node = self._resolve_to_single_node(nodes, key, index, name_token, use_get=use_get)
        if node is not None:
            return node.value
        return None

    @staticmethod
    def _find_node_via_name_token(name_token, elements):
        for node in elements:
            if name_token is node.value.field_token:
                return node
        return None

    def contains_kvpair_element(self, item):
        if not isinstance(item, (str, tuple, Deb822FieldNameToken)):
            return False
        item = cast('ParagraphKey', item)
        try:
            return self.get_kvpair_element(item, use_get=True) is not None
        except AmbiguousDeb822FieldKeyError:
            return True

    def set_kvpair_element(self, key, value):
        key, index, name_token = _unpack_key(key)
        if name_token:
            if name_token is not value.field_token:
                original_nodes = self._kvpair_elements.get(value.field_name)
                original_node = None
                if original_nodes is not None:
                    original_node = self._find_node_via_name_token(name_token, original_nodes)
                if original_node is None:
                    raise ValueError('Key is a Deb822FieldNameToken, but not *the* Deb822FieldNameToken for the value nor the Deb822FieldNameToken for an existing field in the paragraph')
                assert original_nodes is not None
                index = original_nodes.index(original_node)
            key = value.field_name
        else:
            if key != value.field_name:
                raise ValueError('Cannot insert value under a different field value than field name from its Deb822FieldNameToken implies')
            key = value.field_name
        original_nodes = self._kvpair_elements.get(key)
        if original_nodes is None or not original_nodes:
            if index is not None and index != 0:
                msg = 'Cannot replace field ({key}, {index}) as the field does not exist in the first place.  Please index-less key or ({key}, 0) if you want to add the field.'
                raise KeyError(msg.format(key=key, index=index))
            node = self._kvpair_order.append(value)
            if key not in self._kvpair_elements:
                self._kvpair_elements[key] = [node]
            else:
                self._kvpair_elements[key].append(node)
            return
        replace_all = False
        if index is None:
            replace_all = True
            node = original_nodes[0]
            if len(original_nodes) != 1:
                self._kvpair_elements[key] = [node]
        else:
            node = original_nodes[index]
        node.value.parent_element = None
        value.parent_element = self
        node.value = value
        if replace_all and len(original_nodes) != 1:
            for n in original_nodes[1:]:
                n.value.parent_element = None
                self._kvpair_order.remove_node(n)

    def remove_kvpair_element(self, key):
        key, idx, name_token = _unpack_key(key)
        field_list = self._kvpair_elements[key]
        if name_token is None and idx is None:
            for node in field_list:
                node.value.parent_element = None
                self._kvpair_order.remove_node(node)
            del self._kvpair_elements[key]
            return
        if name_token is not None:
            original_node = self._find_node_via_name_token(name_token, field_list)
            if original_node is None:
                msg = 'The field "{key}" is present but key used to access it is not.'
                raise KeyError(msg.format(key=key))
            node = original_node
        else:
            assert idx is not None
            try:
                node = field_list[idx]
            except KeyError:
                msg = 'The field "{key}" is present, but the index "{idx}" was invalid.'
                raise KeyError(msg.format(key=key, idx=idx))
        if len(field_list) == 1:
            del self._kvpair_elements[key]
        else:
            field_list.remove(node)
        node.value.parent_element = None
        self._kvpair_order.remove_node(node)

    def sort_fields(self, key=None):
        """Re-order all fields

        :param key: Provide a key function (same semantics as for sorted).   Keep in mind that
          the module preserve the cases for field names - in generally, callers are recommended
          to use "lower()" to normalize the case.
        """
        if key is None:
            key = default_field_sort_key
        key_impl = key

        def _actual_key(kvpair):
            return key_impl(kvpair.field_name)
        for last_kvpair in reversed(self._kvpair_order):
            last_kvpair.value_element.add_final_newline_if_missing()
            break
        sorted_kvpair_list = sorted(self._kvpair_order, key=_actual_key)
        self._kvpair_order = LinkedList()
        self._kvpair_elements = {}
        self._init_kvpair_fields(sorted_kvpair_list)