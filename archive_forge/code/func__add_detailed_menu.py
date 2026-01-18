import re
import textwrap
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Pattern, Set,
from docutils import nodes, writers
from docutils.nodes import Element, Node, Text
from sphinx import __display_version__, addnodes
from sphinx.domains import IndexEntry
from sphinx.domains.index import IndexDomain
from sphinx.errors import ExtensionError
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.i18n import format_date
from sphinx.writers.latex import collected_footnote
def _add_detailed_menu(name: str) -> None:
    entries = self.node_menus[name]
    if not entries:
        return
    self.body.append('\n%s\n\n' % self.escape(self.node_names[name]))
    self.add_menu_entries(entries)
    for subentry in entries:
        _add_detailed_menu(subentry)