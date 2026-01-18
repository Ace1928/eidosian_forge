import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
from docutils.nodes import Element
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_id, make_refnode
from sphinx.util.typing import OptionSpec
class ReSTDirective(ReSTMarkup):
    """
    Description of a reST directive.
    """

    def handle_signature(self, sig: str, signode: desc_signature) -> str:
        name, args = parse_directive(sig)
        desc_name = f'.. {name}::'
        signode['fullname'] = name.strip()
        signode += addnodes.desc_name(desc_name, desc_name)
        if len(args) > 0:
            signode += addnodes.desc_addname(args, args)
        return name

    def get_index_text(self, objectname: str, name: str) -> str:
        return _('%s (directive)') % name

    def before_content(self) -> None:
        if self.names:
            directives = self.env.ref_context.setdefault('rst:directives', [])
            directives.append(self.names[0])

    def after_content(self) -> None:
        directives = self.env.ref_context.setdefault('rst:directives', [])
        if directives:
            directives.pop()