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
def _walk_doc(docname: str, secnum: Tuple[int, ...]) -> None:
    if docname not in assigned:
        assigned.add(docname)
        doctree = env.get_doctree(docname)
        _walk_doctree(docname, doctree, secnum)