from codecs import open
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta, tzinfo
from os import getenv, path, walk
from time import time
from typing import (Any, DefaultDict, Dict, Generator, Iterable, List, Optional, Set, Tuple,
from uuid import uuid4
from docutils import nodes
from docutils.nodes import Element
from sphinx import addnodes, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.domains.python import pairindextypes
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging, split_index_msg, status_iterator
from sphinx.util.console import bold  # type: ignore
from sphinx.util.i18n import CatalogInfo, docname_to_domain
from sphinx.util.nodes import extract_messages, traverse_translatable_index
from sphinx.util.osutil import canon_path, ensuredir, relpath
from sphinx.util.tags import Tags
from sphinx.util.template import SphinxRenderer
class MsgOrigin:
    """
    Origin holder for Catalog message origin.
    """

    def __init__(self, source: str, line: int) -> None:
        self.source = source
        self.line = line
        self.uid = uuid4().hex