important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class Name(_LeafWithoutNewlines):
    """
    A string. Sometimes it is important to know if the string belongs to a name
    or not.
    """
    type = 'name'
    __slots__ = ()

    def __repr__(self):
        return '<%s: %s@%s,%s>' % (type(self).__name__, self.value, self.line, self.column)

    def is_definition(self, include_setitem=False):
        """
        Returns True if the name is being defined.
        """
        return self.get_definition(include_setitem=include_setitem) is not None

    def get_definition(self, import_name_always=False, include_setitem=False):
        """
        Returns None if there's no definition for a name.

        :param import_name_always: Specifies if an import name is always a
            definition. Normally foo in `from foo import bar` is not a
            definition.
        """
        node = self.parent
        type_ = node.type
        if type_ in ('funcdef', 'classdef'):
            if self == node.name:
                return node
            return None
        if type_ == 'except_clause':
            if self.get_previous_sibling() == 'as':
                return node.parent
            return None
        while node is not None:
            if node.type == 'suite':
                return None
            if node.type in _GET_DEFINITION_TYPES:
                if self in node.get_defined_names(include_setitem):
                    return node
                if import_name_always and node.type in _IMPORTS:
                    return node
                return None
            node = node.parent
        return None