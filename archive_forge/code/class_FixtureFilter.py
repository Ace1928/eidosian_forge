import sys
from typing import List
from pathlib import Path
from parso.tree import search_ancestor
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.imports import goto_import, load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.inference.helpers import infer_call_of_leaf
class FixtureFilter(ParserTreeFilter):

    def _filter(self, names):
        for name in super()._filter(names):
            if name.parent.type == 'import_from':
                imported_names = goto_import(self.parent_context, name)
                if any((self._is_fixture(iname.parent_context, iname.tree_name) for iname in imported_names if iname.tree_name)):
                    yield name
            elif self._is_fixture(self.parent_context, name):
                yield name

    def _is_fixture(self, context, name):
        funcdef = name.parent
        if funcdef.type != 'funcdef':
            return False
        decorated = funcdef.parent
        if decorated.type != 'decorated':
            return False
        decorators = decorated.children[0]
        if decorators.type == 'decorators':
            decorators = decorators.children
        else:
            decorators = [decorators]
        for decorator in decorators:
            dotted_name = decorator.children[1]
            if 'fixture' in dotted_name.get_code():
                if dotted_name.type == 'atom_expr':
                    last_trailer = dotted_name.children[-1]
                    last_leaf = last_trailer.get_last_leaf()
                    if last_leaf == ')':
                        values = infer_call_of_leaf(context, last_leaf, cut_own_trailer=True)
                    else:
                        values = context.infer_node(dotted_name)
                else:
                    values = context.infer_node(dotted_name)
                for value in values:
                    if value.name.get_qualified_names(include_module_names=True) == ('_pytest', 'fixtures', 'fixture'):
                        return True
        return False