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
def _extract_from_template(self) -> None:
    files = list(self._collect_templates())
    files.sort()
    logger.info(bold(__('building [%s]: ') % self.name), nonl=True)
    logger.info(__('targets for %d template files'), len(files))
    extract_translations = self.templates.environment.extract_translations
    for template in status_iterator(files, __('reading templates... '), 'purple', len(files), self.app.verbosity):
        try:
            with open(template, encoding='utf-8') as f:
                context = f.read()
            for line, _meth, msg in extract_translations(context):
                origin = MsgOrigin(template, line)
                self.catalogs['sphinx'].add(msg, origin)
        except Exception as exc:
            raise ThemeError('%s: %r' % (template, exc)) from exc