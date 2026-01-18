import collections.abc
import dataclasses
import inspect
from typing import Any
from typing import Callable
from typing import Collection
from typing import final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from .._code import getfslineno
from ..compat import ascii_escaped
from ..compat import NOTSET
from ..compat import NotSetType
from _pytest.config import Config
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import MARKED_FIXTURE
from _pytest.outcomes import fail
from _pytest.warning_types import PytestUnknownMarkWarning
@final
class MarkGenerator:
    """Factory for :class:`MarkDecorator` objects - exposed as
    a ``pytest.mark`` singleton instance.

    Example::

         import pytest


         @pytest.mark.slowtest
         def test_function():
             pass

    applies a 'slowtest' :class:`Mark` on ``test_function``.
    """
    if TYPE_CHECKING:
        skip: _SkipMarkDecorator
        skipif: _SkipifMarkDecorator
        xfail: _XfailMarkDecorator
        parametrize: _ParametrizeMarkDecorator
        usefixtures: _UsefixturesMarkDecorator
        filterwarnings: _FilterwarningsMarkDecorator

    def __init__(self, *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        self._config: Optional[Config] = None
        self._markers: Set[str] = set()

    def __getattr__(self, name: str) -> MarkDecorator:
        """Generate a new :class:`MarkDecorator` with the given name."""
        if name[0] == '_':
            raise AttributeError('Marker name must NOT start with underscore')
        if self._config is not None:
            if name not in self._markers:
                for line in self._config.getini('markers'):
                    marker = line.split(':')[0].split('(')[0].strip()
                    self._markers.add(marker)
            if name not in self._markers:
                if self._config.option.strict_markers or self._config.option.strict:
                    fail(f'{name!r} not found in `markers` configuration option', pytrace=False)
                if name in ['parameterize', 'parametrise', 'parameterise']:
                    __tracebackhide__ = True
                    fail(f"Unknown '{name}' mark, did you mean 'parametrize'?")
                warnings.warn('Unknown pytest.mark.%s - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html' % name, PytestUnknownMarkWarning, 2)
        return MarkDecorator(Mark(name, (), {}, _ispytest=True), _ispytest=True)