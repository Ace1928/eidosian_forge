import concurrent.futures
import functools
import posixpath
import re
import sys
import time
from os import path
from types import ModuleType
from typing import IO, Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urlsplit, urlunsplit
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.utils import Reporter, relative_path
import sphinx
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders.html import INVENTORY_FILENAME
from sphinx.config import Config
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError
from sphinx.locale import _, __
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging, requests
from sphinx.util.docutils import CustomReSTDispatcher, SphinxRole
from sphinx.util.inventory import InventoryFile
from sphinx.util.typing import Inventory, InventoryItem, RoleFunction
def _resolve_reference_in_domain(env: BuildEnvironment, inv_name: Optional[str], inventory: Inventory, honor_disabled_refs: bool, domain: Domain, objtypes: List[str], node: pending_xref, contnode: TextElement) -> Optional[Element]:
    if domain.name == 'std' and 'cmdoption' in objtypes:
        objtypes.append('option')
    if domain.name == 'py' and 'attribute' in objtypes:
        objtypes.append('method')
    objtypes = ['{}:{}'.format(domain.name, t) for t in objtypes]
    if honor_disabled_refs:
        disabled = env.config.intersphinx_disabled_reftypes
        objtypes = [o for o in objtypes if o not in disabled]
    res = _resolve_reference_in_domain_by_target(inv_name, inventory, domain, objtypes, node['reftarget'], node, contnode)
    if res is not None:
        return res
    full_qualified_name = domain.get_full_qualified_name(node)
    if full_qualified_name is None:
        return None
    return _resolve_reference_in_domain_by_target(inv_name, inventory, domain, objtypes, full_qualified_name, node, contnode)