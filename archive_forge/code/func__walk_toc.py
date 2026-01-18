from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union, cast
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.environment.adapters.toctree import TocTree
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.locale import __
from sphinx.transforms import SphinxContentsFilter
from sphinx.util import logging, url_re
def _walk_toc(node: Element, secnums: Dict, depth: int, titlenode: Optional[nodes.title]=None) -> None:
    for subnode in node.children:
        if isinstance(subnode, nodes.bullet_list):
            numstack.append(0)
            _walk_toc(subnode, secnums, depth - 1, titlenode)
            numstack.pop()
            titlenode = None
        elif isinstance(subnode, nodes.list_item):
            _walk_toc(subnode, secnums, depth, titlenode)
            titlenode = None
        elif isinstance(subnode, addnodes.only):
            _walk_toc(subnode, secnums, depth, titlenode)
            titlenode = None
        elif isinstance(subnode, addnodes.compact_paragraph):
            if 'skip_section_number' in subnode:
                continue
            numstack[-1] += 1
            reference = cast(nodes.reference, subnode[0])
            if depth > 0:
                number = list(numstack)
                secnums[reference['anchorname']] = tuple(numstack)
            else:
                number = None
                secnums[reference['anchorname']] = None
            reference['secnumber'] = number
            if titlenode:
                titlenode['secnumber'] = number
                titlenode = None
        elif isinstance(subnode, addnodes.toctree):
            _walk_toctree(subnode, depth)