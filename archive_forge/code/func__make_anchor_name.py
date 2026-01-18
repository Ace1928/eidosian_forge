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
def _make_anchor_name(ids: List[str], num_entries: List[int]) -> str:
    if not num_entries[0]:
        anchorname = ''
    else:
        anchorname = '#' + ids[0]
    num_entries[0] += 1
    return anchorname