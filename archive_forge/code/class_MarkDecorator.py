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
@dataclasses.dataclass
class MarkDecorator:
    """A decorator for applying a mark on test functions and classes.

    ``MarkDecorators`` are created with ``pytest.mark``::

        mark1 = pytest.mark.NAME  # Simple MarkDecorator
        mark2 = pytest.mark.NAME(name1=value)  # Parametrized MarkDecorator

    and can then be applied as decorators to test functions::

        @mark2
        def test_function():
            pass

    When a ``MarkDecorator`` is called, it does the following:

    1. If called with a single class as its only positional argument and no
       additional keyword arguments, it attaches the mark to the class so it
       gets applied automatically to all test cases found in that class.

    2. If called with a single function as its only positional argument and
       no additional keyword arguments, it attaches the mark to the function,
       containing all the arguments already stored internally in the
       ``MarkDecorator``.

    3. When called in any other case, it returns a new ``MarkDecorator``
       instance with the original ``MarkDecorator``'s content updated with
       the arguments passed to this call.

    Note: The rules above prevent a ``MarkDecorator`` from storing only a
    single function or class reference as its positional argument with no
    additional keyword or positional arguments. You can work around this by
    using `with_args()`.
    """
    mark: Mark

    def __init__(self, mark: Mark, *, _ispytest: bool=False) -> None:
        """:meta private:"""
        check_ispytest(_ispytest)
        self.mark = mark

    @property
    def name(self) -> str:
        """Alias for mark.name."""
        return self.mark.name

    @property
    def args(self) -> Tuple[Any, ...]:
        """Alias for mark.args."""
        return self.mark.args

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Alias for mark.kwargs."""
        return self.mark.kwargs

    @property
    def markname(self) -> str:
        """:meta private:"""
        return self.name

    def with_args(self, *args: object, **kwargs: object) -> 'MarkDecorator':
        """Return a MarkDecorator with extra arguments added.

        Unlike calling the MarkDecorator, with_args() can be used even
        if the sole argument is a callable/class.
        """
        mark = Mark(self.name, args, kwargs, _ispytest=True)
        return MarkDecorator(self.mark.combined_with(mark), _ispytest=True)

    @overload
    def __call__(self, arg: Markable) -> Markable:
        pass

    @overload
    def __call__(self, *args: object, **kwargs: object) -> 'MarkDecorator':
        pass

    def __call__(self, *args: object, **kwargs: object):
        """Call the MarkDecorator."""
        if args and (not kwargs):
            func = args[0]
            is_class = inspect.isclass(func)
            if len(args) == 1 and (istestfunc(func) or is_class):
                store_mark(func, self.mark, stacklevel=3)
                return func
        return self.with_args(*args, **kwargs)