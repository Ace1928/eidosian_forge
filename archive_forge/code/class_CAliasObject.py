import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class CAliasObject(ObjectDescription):
    option_spec: OptionSpec = {'maxdepth': directives.nonnegative_int, 'noroot': directives.flag}

    def run(self) -> List[Node]:
        """
        On purpose this doesn't call the ObjectDescription version, but is based on it.
        Each alias signature may expand into multiple real signatures if 'noroot'.
        The code is therefore based on the ObjectDescription version.
        """
        if ':' in self.name:
            self.domain, self.objtype = self.name.split(':', 1)
        else:
            self.domain, self.objtype = ('', self.name)
        node = addnodes.desc()
        node.document = self.state.document
        node['domain'] = self.domain
        node['objtype'] = node['desctype'] = self.objtype
        node['noindex'] = True
        self.names: List[str] = []
        aliasOptions = {'maxdepth': self.options.get('maxdepth', 1), 'noroot': 'noroot' in self.options}
        if aliasOptions['noroot'] and aliasOptions['maxdepth'] == 1:
            logger.warning("Error in C alias declaration. Requested 'noroot' but 'maxdepth' 1. When skipping the root declaration, need 'maxdepth' 0 for infinite or at least 2.", location=self.get_location())
        for sig in self.get_signatures():
            node.append(AliasNode(sig, aliasOptions, self.state.document, env=self.env))
        return [node]