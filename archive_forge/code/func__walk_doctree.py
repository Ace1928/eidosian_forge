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
def _walk_doctree(docname: str, doctree: Element, secnum: Tuple[int, ...]) -> None:
    nonlocal generated_docnames
    for subnode in doctree.children:
        if isinstance(subnode, nodes.section):
            next_secnum = get_section_number(docname, subnode)
            if next_secnum:
                _walk_doctree(docname, subnode, next_secnum)
            else:
                _walk_doctree(docname, subnode, secnum)
        elif isinstance(subnode, addnodes.toctree):
            for _title, subdocname in subnode['entries']:
                if url_re.match(subdocname) or subdocname == 'self':
                    continue
                if subdocname in generated_docnames:
                    continue
                _walk_doc(subdocname, secnum)
        elif isinstance(subnode, nodes.Element):
            figtype = get_figtype(subnode)
            if figtype and subnode['ids']:
                register_fignumber(docname, secnum, figtype, subnode)
            _walk_doctree(docname, subnode, secnum)