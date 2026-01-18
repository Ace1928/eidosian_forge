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
class ConfigGroupXRefRole(XRefRole):
    """Handles :oslo.config:group: roles pointing to configuration groups."""

    def __init__(self):
        super(ConfigGroupXRefRole, self).__init__(warn_dangling=True)

    def process_link(self, env, refnode, has_explicit_title, title, target):
        return (target, target)