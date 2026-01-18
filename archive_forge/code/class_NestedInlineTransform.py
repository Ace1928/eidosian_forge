from typing import Any, Dict, Iterable, cast
from docutils import nodes
from docutils.nodes import Element, TextElement
from docutils.writers.manpage import Translator as BaseTranslator
from docutils.writers.manpage import Writer
from sphinx import addnodes
from sphinx.builders import Builder
from sphinx.locale import _, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.i18n import format_date
from sphinx.util.nodes import NodeMatcher
class NestedInlineTransform:
    """
    Flatten nested inline nodes:

    Before:
        <strong>foo=<emphasis>1</emphasis>
        &bar=<emphasis>2</emphasis></strong>
    After:
        <strong>foo=</strong><emphasis>var</emphasis>
        <strong>&bar=</strong><emphasis>2</emphasis>
    """

    def __init__(self, document: nodes.document) -> None:
        self.document = document

    def apply(self, **kwargs: Any) -> None:
        matcher = NodeMatcher(nodes.literal, nodes.emphasis, nodes.strong)
        for node in list(self.document.findall(matcher)):
            if any((matcher(subnode) for subnode in node)):
                pos = node.parent.index(node)
                for subnode in reversed(list(node)):
                    node.remove(subnode)
                    if matcher(subnode):
                        node.parent.insert(pos + 1, subnode)
                    else:
                        newnode = node.__class__('', '', subnode, **node.attributes)
                        node.parent.insert(pos + 1, newnode)
                if not len(node):
                    node.parent.remove(node)