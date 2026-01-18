import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class CPPXRefRole(XRefRole):

    def process_link(self, env: BuildEnvironment, refnode: Element, has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        refnode.attributes.update(env.ref_context)
        if not has_explicit_title:
            title = anon_identifier_re.sub('[anonymous]', str(title))
        if refnode['reftype'] == 'any':
            if not has_explicit_title and title.endswith('()'):
                title = title[:-2]
            if target.endswith('()'):
                target = target[:-2]
        if not has_explicit_title:
            target = target.lstrip('~')
            if title[:1] == '~':
                title = title[1:]
                dcolon = title.rfind('::')
                if dcolon != -1:
                    title = title[dcolon + 2:]
        return (title, target)