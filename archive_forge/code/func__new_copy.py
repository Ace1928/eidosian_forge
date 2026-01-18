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
def _new_copy(self: Element) -> Element:
    """monkey-patch Element.copy to copy the rawsource and line
    for docutils-0.16 or older versions.

    refs: https://sourceforge.net/p/docutils/patches/165/
    """
    newnode = self.__class__(self.rawsource, **self.attributes)
    if isinstance(self, nodes.Element):
        newnode.source = self.source
        newnode.line = self.line
    return newnode