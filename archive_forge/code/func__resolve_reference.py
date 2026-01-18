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
def _resolve_reference(env: BuildEnvironment, inv_name: Optional[str], inventory: Inventory, honor_disabled_refs: bool, node: pending_xref, contnode: TextElement) -> Optional[Element]:
    honor_disabled_refs = honor_disabled_refs and inv_name is None
    if honor_disabled_refs and '*' in env.config.intersphinx_disabled_reftypes:
        return None
    typ = node['reftype']
    if typ == 'any':
        for domain_name, domain in env.domains.items():
            if honor_disabled_refs and domain_name + ':*' in env.config.intersphinx_disabled_reftypes:
                continue
            objtypes = list(domain.object_types)
            res = _resolve_reference_in_domain(env, inv_name, inventory, honor_disabled_refs, domain, objtypes, node, contnode)
            if res is not None:
                return res
        return None
    else:
        domain_name = node.get('refdomain')
        if not domain_name:
            return None
        if honor_disabled_refs and domain_name + ':*' in env.config.intersphinx_disabled_reftypes:
            return None
        domain = env.get_domain(domain_name)
        objtypes = domain.objtypes_for_role(typ)
        if not objtypes:
            return None
        return _resolve_reference_in_domain(env, inv_name, inventory, honor_disabled_refs, domain, objtypes, node, contnode)