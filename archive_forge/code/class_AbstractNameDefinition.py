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
class AbstractNameDefinition:
    start_pos: Optional[Tuple[int, int]] = None
    string_name: str
    parent_context = None
    tree_name = None
    is_value_name = True
    "\n    Used for the Jedi API to know if it's a keyword or an actual name.\n    "

    @abstractmethod
    def infer(self):
        raise NotImplementedError

    @abstractmethod
    def goto(self):
        return {self}

    def get_qualified_names(self, include_module_names=False):
        qualified_names = self._get_qualified_names()
        if qualified_names is None or not include_module_names:
            return qualified_names
        module_names = self.get_root_context().string_names
        if module_names is None:
            return None
        return module_names + qualified_names

    def _get_qualified_names(self):
        return None

    def get_root_context(self):
        return self.parent_context.get_root_context()

    def get_public_name(self):
        return self.string_name

    def __repr__(self):
        if self.start_pos is None:
            return '<%s: string_name=%s>' % (self.__class__.__name__, self.string_name)
        return '<%s: string_name=%s start_pos=%s>' % (self.__class__.__name__, self.string_name, self.start_pos)

    def is_import(self):
        return False

    def py__doc__(self):
        return ''

    @property
    def api_type(self):
        return self.parent_context.api_type

    def get_defining_qualified_value(self):
        """
        Returns either None or the value that is public and qualified. Won't
        return a function, because a name in a function is never public.
        """
        return None