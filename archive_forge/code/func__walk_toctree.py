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
def _walk_toctree(toctreenode: addnodes.toctree, depth: int) -> None:
    if depth == 0:
        return
    for _title, ref in toctreenode['entries']:
        if url_re.match(ref) or ref == 'self':
            continue
        elif ref in assigned:
            logger.warning(__('%s is already assigned section numbers (nested numbered toctree?)'), ref, location=toctreenode, type='toc', subtype='secnum')
        elif ref in env.tocs:
            secnums: Dict[str, Tuple[int, ...]] = {}
            env.toc_secnumbers[ref] = secnums
            assigned.add(ref)
            _walk_toc(env.tocs[ref], secnums, depth, env.titles.get(ref))
            if secnums != old_secnumbers.get(ref):
                rewrite_needed.append(ref)