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
def _resolve_xref_inner(self, env: BuildEnvironment, fromdocname: str, builder: Builder, typ: str, target: str, node: pending_xref, contnode: Element) -> Tuple[Optional[Element], Optional[str]]:
    parser = DefinitionParser(target, location=node, config=env.config)
    try:
        name = parser.parse_xref_object()
    except DefinitionError as e:
        logger.warning('Unparseable C cross-reference: %r\n%s', target, e, location=node)
        return (None, None)
    parentKey: LookupKey = node.get('c:parent_key', None)
    rootSymbol = self.data['root_symbol']
    if parentKey:
        parentSymbol: Symbol = rootSymbol.direct_lookup(parentKey)
        if not parentSymbol:
            print('Target: ', target)
            print('ParentKey: ', parentKey)
            print(rootSymbol.dump(1))
        assert parentSymbol
    else:
        parentSymbol = rootSymbol
    s = parentSymbol.find_declaration(name, typ, matchSelf=True, recurseInAnon=True)
    if s is None or s.declaration is None:
        return (None, None)
    declaration = s.declaration
    displayName = name.get_display_string()
    docname = s.docname
    assert docname
    return (make_refnode(builder, fromdocname, docname, declaration.get_newest_id(), contnode, displayName), declaration.objectType)