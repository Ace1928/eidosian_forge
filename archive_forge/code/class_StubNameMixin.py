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
class StubNameMixin:

    def py__doc__(self):
        from jedi.inference.gradual.conversion import convert_names
        names = [self]
        if self.api_type == 'statement' and '=' in self.tree_name.get_definition().children:
            names = [v.name for v in self.infer()]
        names = convert_names(names, prefer_stub_to_compiled=False)
        if self in names:
            return super().py__doc__()
        else:
            return _merge_name_docs(names)