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
class AliasNode(nodes.Element):

    def __init__(self, sig: str, aliasOptions: dict, document: Any, env: 'BuildEnvironment'=None, parentKey: LookupKey=None) -> None:
        super().__init__()
        self.sig = sig
        self.aliasOptions = aliasOptions
        self.document = document
        if env is not None:
            if 'c:parent_symbol' not in env.temp_data:
                root = env.domaindata['c']['root_symbol']
                env.temp_data['c:parent_symbol'] = root
                env.ref_context['c:parent_key'] = root.get_lookup_key()
            self.parentKey = env.ref_context['c:parent_key']
        else:
            assert parentKey is not None
            self.parentKey = parentKey

    def copy(self) -> 'AliasNode':
        return self.__class__(self.sig, self.aliasOptions, self.document, env=None, parentKey=self.parentKey)