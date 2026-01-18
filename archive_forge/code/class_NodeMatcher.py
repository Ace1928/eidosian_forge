import re
import unicodedata
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple, Type,
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging
class NodeMatcher:
    """A helper class for Node.findall().

    It checks that the given node is an instance of the specified node-classes and
    has the specified node-attributes.

    For example, following example searches ``reference`` node having ``refdomain``
    and ``reftype`` attributes::

        matcher = NodeMatcher(nodes.reference, refdomain='std', reftype='citation')
        doctree.findall(matcher)
        # => [<reference ...>, <reference ...>, ...]

    A special value ``typing.Any`` matches any kind of node-attributes.  For example,
    following example searches ``reference`` node having ``refdomain`` attributes::

        from typing import Any
        matcher = NodeMatcher(nodes.reference, refdomain=Any)
        doctree.findall(matcher)
        # => [<reference ...>, <reference ...>, ...]
    """

    def __init__(self, *node_classes: Type[Node], **attrs: Any) -> None:
        self.classes = node_classes
        self.attrs = attrs

    def match(self, node: Node) -> bool:
        try:
            if self.classes and (not isinstance(node, self.classes)):
                return False
            if self.attrs:
                if not isinstance(node, nodes.Element):
                    return False
                for key, value in self.attrs.items():
                    if key not in node:
                        return False
                    elif value is Any:
                        continue
                    elif node.get(key) != value:
                        return False
            return True
        except Exception:
            return False

    def __call__(self, node: Node) -> bool:
        return self.match(node)