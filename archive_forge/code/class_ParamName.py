import re
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from jedi import settings
from jedi import debug
from jedi.inference.utils import unite
from jedi.cache import memoize_method
from jedi.inference.compiled.mixed import MixedName
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.conversion import convert_names, convert_values
from jedi.inference.base_value import ValueSet, HasNoContext
from jedi.api.keywords import KeywordName
from jedi.api import completion_cache
from jedi.api.helpers import filter_follow_imports
class ParamName(Name):

    def infer_default(self):
        """
        Returns default values like the ``1`` of ``def foo(x=1):``.

        :rtype: list of :class:`.Name`
        """
        return _values_to_definitions(self._name.infer_default())

    def infer_annotation(self, **kwargs):
        """
        :param execute_annotation: Default True; If False, values are not
            executed and classes are returned instead of instances.
        :rtype: list of :class:`.Name`
        """
        return _values_to_definitions(self._name.infer_annotation(ignore_stars=True, **kwargs))

    def to_string(self):
        """
        Returns a simple representation of a param, like
        ``f: Callable[..., Any]``.

        :rtype: str
        """
        return self._name.to_string()

    @property
    def kind(self):
        """
        Returns an enum instance of :mod:`inspect`'s ``Parameter`` enum.

        :rtype: :py:attr:`inspect.Parameter.kind`
        """
        return self._name.get_kind()