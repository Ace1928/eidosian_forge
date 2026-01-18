from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging, url_re
from sphinx.util.matching import Matcher
from sphinx.util.nodes import clean_astext, process_only_nodes
def _toctree_prune(self, node: Element, depth: int, maxdepth: int, collapse: bool=False) -> None:
    """Utility: Cut a TOC at a specified depth."""
    for subnode in node.children[:]:
        if isinstance(subnode, (addnodes.compact_paragraph, nodes.list_item)):
            self._toctree_prune(subnode, depth, maxdepth, collapse)
        elif isinstance(subnode, nodes.bullet_list):
            if maxdepth > 0 and depth > maxdepth:
                subnode.parent.replace(subnode, [])
            elif collapse and depth > 1 and ('iscurrent' not in subnode.parent):
                subnode.parent.remove(subnode)
            else:
                self._toctree_prune(subnode, depth + 1, maxdepth, collapse)