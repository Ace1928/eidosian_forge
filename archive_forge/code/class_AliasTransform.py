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
class AliasTransform(SphinxTransform):
    default_priority = ReferencesResolver.default_priority - 1

    def _render_symbol(self, s: Symbol, maxdepth: int, skipThis: bool, aliasOptions: dict, renderOptions: dict, document: Any) -> List[Node]:
        if maxdepth == 0:
            recurse = True
        elif maxdepth == 1:
            recurse = False
        else:
            maxdepth -= 1
            recurse = True
        nodes: List[Node] = []
        if not skipThis:
            signode = addnodes.desc_signature('', '')
            nodes.append(signode)
            s.declaration.describe_signature(signode, 'markName', self.env, renderOptions)
        if recurse:
            if skipThis:
                childContainer: Union[List[Node], addnodes.desc] = nodes
            else:
                content = addnodes.desc_content()
                desc = addnodes.desc()
                content.append(desc)
                desc.document = document
                desc['domain'] = 'c'
                desc['objtype'] = desc['desctype'] = 'alias'
                desc['noindex'] = True
                childContainer = desc
            for sChild in s.children:
                if sChild.declaration is None:
                    continue
                childNodes = self._render_symbol(sChild, maxdepth=maxdepth, skipThis=False, aliasOptions=aliasOptions, renderOptions=renderOptions, document=document)
                childContainer.extend(childNodes)
            if not skipThis and len(desc.children) != 0:
                nodes.append(content)
        return nodes

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.findall(AliasNode):
            sig = node.sig
            parentKey = node.parentKey
            try:
                parser = DefinitionParser(sig, location=node, config=self.env.config)
                name = parser.parse_xref_object()
            except DefinitionError as e:
                logger.warning(e, location=node)
                name = None
            if name is None:
                signode = addnodes.desc_signature(sig, '')
                signode.clear()
                signode += addnodes.desc_name(sig, sig)
                node.replace_self(signode)
                continue
            rootSymbol: Symbol = self.env.domains['c'].data['root_symbol']
            parentSymbol: Symbol = rootSymbol.direct_lookup(parentKey)
            if not parentSymbol:
                print('Target: ', sig)
                print('ParentKey: ', parentKey)
                print(rootSymbol.dump(1))
            assert parentSymbol
            s = parentSymbol.find_declaration(name, 'any', matchSelf=True, recurseInAnon=True)
            if s is None:
                signode = addnodes.desc_signature(sig, '')
                node.append(signode)
                signode.clear()
                signode += addnodes.desc_name(sig, sig)
                logger.warning("Could not find C declaration for alias '%s'." % name, location=node)
                node.replace_self(signode)
                continue
            if not node.aliasOptions['noroot'] and s.declaration is None:
                signode = addnodes.desc_signature(sig, '')
                node.append(signode)
                signode.clear()
                signode += addnodes.desc_name(sig, sig)
                logger.warning("Can not render C declaration for alias '%s'. No such declaration." % name, location=node)
                node.replace_self(signode)
                continue
            nodes = self._render_symbol(s, maxdepth=node.aliasOptions['maxdepth'], skipThis=node.aliasOptions['noroot'], aliasOptions=node.aliasOptions, renderOptions={}, document=node.document)
            node.replace_self(nodes)