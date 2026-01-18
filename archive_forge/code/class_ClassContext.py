from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi.inference.filters import ParserTreeFilter, MergedFilter, \
from jedi.inference.names import AnonymousParamName, TreeNameDefinition
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.parser_utils import get_parent_scope
from jedi import debug
from jedi import parser_utils
class ClassContext(TreeContextMixin, ValueContext):

    def get_filters(self, until_position=None, origin_scope=None):
        yield self.get_global_filter(until_position, origin_scope)

    def get_global_filter(self, until_position=None, origin_scope=None):
        return ParserTreeFilter(parent_context=self, until_position=until_position, origin_scope=origin_scope)