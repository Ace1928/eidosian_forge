from abc import abstractmethod
from inspect import Parameter
from typing import Optional, Tuple
from parso.tree import search_ancestor
from jedi.parser_utils import find_statement_documentation, clean_scope_docstring
from jedi.inference.utils import unite
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.cache import inference_state_method_cache
from jedi.inference import docstrings
from jedi.cache import memoize_method
from jedi.inference.helpers import deep_ast_copy, infer_call_of_leaf
from jedi.plugins import plugin_manager
class ValueNameMixin:

    def infer(self):
        return ValueSet([self._value])

    def py__doc__(self):
        doc = self._value.py__doc__()
        if not doc and self._value.is_stub():
            from jedi.inference.gradual.conversion import convert_names
            names = convert_names([self], prefer_stub_to_compiled=False)
            if self not in names:
                return _merge_name_docs(names)
        return doc

    def _get_qualified_names(self):
        return self._value.get_qualified_names()

    def get_root_context(self):
        if self.parent_context is None:
            return self._value.as_context()
        return super().get_root_context()

    def get_defining_qualified_value(self):
        context = self.parent_context
        if context is not None and (context.is_module() or context.is_class()):
            return self.parent_context.get_value()
        return None

    @property
    def api_type(self):
        return self._value.api_type