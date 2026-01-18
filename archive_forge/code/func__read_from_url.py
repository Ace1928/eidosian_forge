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
def _read_from_url(url: str, config: Optional[Config]=None) -> IO:
    """Reads data from *url* with an HTTP *GET*.

    This function supports fetching from resources which use basic HTTP auth as
    laid out by RFC1738 § 3.1. See § 5 for grammar definitions for URLs.

    .. seealso:

       https://www.ietf.org/rfc/rfc1738.txt

    :param url: URL of an HTTP resource
    :type url: ``str``

    :return: data read from resource described by *url*
    :rtype: ``file``-like object
    """
    r = requests.get(url, stream=True, config=config, timeout=config.intersphinx_timeout)
    r.raise_for_status()
    r.raw.url = r.url
    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    return r.raw