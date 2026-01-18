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
class BaseSignature(Name):
    """
    These signatures are returned by :meth:`BaseName.get_signatures`
    calls.
    """

    def __init__(self, inference_state, signature):
        super().__init__(inference_state, signature.name)
        self._signature = signature

    @property
    def params(self):
        """
        Returns definitions for all parameters that a signature defines.
        This includes stuff like ``*args`` and ``**kwargs``.

        :rtype: list of :class:`.ParamName`
        """
        return [ParamName(self._inference_state, n) for n in self._signature.get_param_names(resolve_stars=True)]

    def to_string(self):
        """
        Returns a text representation of the signature. This could for example
        look like ``foo(bar, baz: int, **kwargs)``.

        :rtype: str
        """
        return self._signature.to_string()