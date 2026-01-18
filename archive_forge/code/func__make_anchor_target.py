from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
import oslo_i18n
from sphinx import addnodes
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain
from sphinx.domains import ObjType
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_refnode
from sphinx.util.nodes import nested_parse_with_titles
from oslo_config import cfg
from oslo_config import generator
def _make_anchor_target(group_name, option_name):
    target = '%s.%s' % (cfg._normalize_group_name(group_name), option_name.lower())
    return target