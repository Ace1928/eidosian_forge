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
def _collect_templates(self) -> Set[str]:
    template_files = set()
    for template_path in self.config.templates_path:
        tmpl_abs_path = path.join(self.app.srcdir, template_path)
        for dirpath, _dirs, files in walk(tmpl_abs_path):
            for fn in files:
                if fn.endswith('.html'):
                    filename = canon_path(path.join(dirpath, fn))
                    template_files.add(filename)
    return template_files