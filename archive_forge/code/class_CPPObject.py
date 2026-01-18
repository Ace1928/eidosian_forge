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
class CPPObject(ObjectDescription[ASTDeclaration]):
    """Description of a C++ language object."""
    doc_field_types: List[Field] = [GroupedField('template parameter', label=_('Template Parameters'), names=('tparam', 'template parameter'), can_collapse=True)]
    option_spec: OptionSpec = {'noindexentry': directives.flag, 'nocontentsentry': directives.flag, 'tparam-line-spec': directives.flag}

    def _add_enumerator_to_parent(self, ast: ASTDeclaration) -> None:
        assert ast.objectType == 'enumerator'
        symbol = ast.symbol
        assert symbol
        assert symbol.identOrOp is not None
        assert symbol.templateParams is None
        assert symbol.templateArgs is None
        parentSymbol = symbol.parent
        assert parentSymbol
        if parentSymbol.parent is None:
            return
        parentDecl = parentSymbol.declaration
        if parentDecl is None:
            return
        if parentDecl.objectType != 'enum':
            return
        if parentDecl.directiveType != 'enum':
            return
        targetSymbol = parentSymbol.parent
        s = targetSymbol.find_identifier(symbol.identOrOp, matchSelf=False, recurseInAnon=True, searchInSiblings=False)
        if s is not None:
            return
        declClone = symbol.declaration.clone()
        declClone.enumeratorScopedSymbol = symbol
        Symbol(parent=targetSymbol, identOrOp=symbol.identOrOp, templateParams=None, templateArgs=None, declaration=declClone, docname=self.env.docname, line=self.get_source_info()[1])

    def add_target_and_index(self, ast: ASTDeclaration, sig: str, signode: TextElement) -> None:
        ids = []
        for i in range(1, _max_id + 1):
            try:
                id = ast.get_id(version=i)
                ids.append(id)
            except NoOldIdError:
                assert i < _max_id
        ids = list(reversed(ids))
        newestId = ids[0]
        assert newestId
        if not re.compile('^[a-zA-Z0-9_]*$').match(newestId):
            logger.warning('Index id generation for C++ object "%s" failed, please report as bug (id=%s).', ast, newestId, location=self.get_location())
        name = ast.symbol.get_full_nested_name().get_display_string().lstrip(':')
        isInConcept = False
        s = ast.symbol.parent
        while s is not None:
            decl = s.declaration
            s = s.parent
            if decl is None:
                continue
            if decl.objectType == 'concept':
                isInConcept = True
                break
        if not isInConcept and 'noindexentry' not in self.options:
            strippedName = name
            for prefix in self.env.config.cpp_index_common_prefix:
                if name.startswith(prefix):
                    strippedName = strippedName[len(prefix):]
                    break
            indexText = self.get_index_text(strippedName)
            self.indexnode['entries'].append(('single', indexText, newestId, '', None))
        if newestId not in self.state.document.ids:
            names = self.env.domaindata['cpp']['names']
            if name not in names:
                names[name] = ast.symbol.docname
            assert newestId
            signode['ids'].append(newestId)
            for id in ids[1:]:
                if not id:
                    continue
                if id not in self.state.document.ids:
                    signode['ids'].append(id)
            self.state.document.note_explicit_target(signode)

    @property
    def object_type(self) -> str:
        raise NotImplementedError()

    @property
    def display_object_type(self) -> str:
        return self.object_type

    def get_index_text(self, name: str) -> str:
        return _('%s (C++ %s)') % (name, self.display_object_type)

    def parse_definition(self, parser: DefinitionParser) -> ASTDeclaration:
        return parser.parse_declaration(self.object_type, self.objtype)

    def describe_signature(self, signode: desc_signature, ast: ASTDeclaration, options: Dict) -> None:
        ast.describe_signature(signode, 'lastIsName', self.env, options)

    def run(self) -> List[Node]:
        env = self.state.document.settings.env
        if 'cpp:parent_symbol' not in env.temp_data:
            root = env.domaindata['cpp']['root_symbol']
            env.temp_data['cpp:parent_symbol'] = root
            env.ref_context['cpp:parent_key'] = root.get_lookup_key()
        parentSymbol = env.temp_data['cpp:parent_symbol']
        parentDecl = parentSymbol.declaration
        if parentDecl is not None and parentDecl.objectType == 'function':
            msg = 'C++ declarations inside functions are not supported. Parent function: {}\nDirective name: {}\nDirective arg: {}'
            logger.warning(msg.format(str(parentSymbol.get_full_nested_name()), self.name, self.arguments[0]), location=self.get_location())
            name = _make_phony_error_name()
            symbol = parentSymbol.add_name(name)
            env.temp_data['cpp:last_symbol'] = symbol
            return []
        env.temp_data['cpp:last_symbol'] = None
        return super().run()

    def handle_signature(self, sig: str, signode: desc_signature) -> ASTDeclaration:
        parentSymbol: Symbol = self.env.temp_data['cpp:parent_symbol']
        parser = DefinitionParser(sig, location=signode, config=self.env.config)
        try:
            ast = self.parse_definition(parser)
            parser.assert_end()
        except DefinitionError as e:
            logger.warning(e, location=signode)
            name = _make_phony_error_name()
            symbol = parentSymbol.add_name(name)
            self.env.temp_data['cpp:last_symbol'] = symbol
            raise ValueError from e
        try:
            symbol = parentSymbol.add_declaration(ast, docname=self.env.docname, line=self.get_source_info()[1])
            assert symbol.siblingAbove is None
            assert symbol.siblingBelow is None
            symbol.siblingAbove = self.env.temp_data['cpp:last_symbol']
            if symbol.siblingAbove is not None:
                assert symbol.siblingAbove.siblingBelow is None
                symbol.siblingAbove.siblingBelow = symbol
            self.env.temp_data['cpp:last_symbol'] = symbol
        except _DuplicateSymbolError as e:
            self.env.temp_data['cpp:last_symbol'] = e.symbol
            msg = __("Duplicate C++ declaration, also defined at %s:%s.\nDeclaration is '.. cpp:%s:: %s'.")
            msg = msg % (e.symbol.docname, e.symbol.line, self.display_object_type, sig)
            logger.warning(msg, location=signode)
        if ast.objectType == 'enumerator':
            self._add_enumerator_to_parent(ast)
        options = dict(self.options)
        options['tparam-line-spec'] = 'tparam-line-spec' in self.options
        self.describe_signature(signode, ast, options)
        return ast

    def before_content(self) -> None:
        lastSymbol: Symbol = self.env.temp_data['cpp:last_symbol']
        assert lastSymbol
        self.oldParentSymbol = self.env.temp_data['cpp:parent_symbol']
        self.oldParentKey: LookupKey = self.env.ref_context['cpp:parent_key']
        self.env.temp_data['cpp:parent_symbol'] = lastSymbol
        self.env.ref_context['cpp:parent_key'] = lastSymbol.get_lookup_key()

    def after_content(self) -> None:
        self.env.temp_data['cpp:parent_symbol'] = self.oldParentSymbol
        self.env.ref_context['cpp:parent_key'] = self.oldParentKey