from pprint import pformat
import re
from types import TracebackType
from typing import Any
from typing import Callable
from typing import final
from typing import Generator
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import warnings
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.outcomes import Exit
from _pytest.outcomes import fail
@final
class WarningsChecker(WarningsRecorder):

    def __init__(self, expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...]]=Warning, match_expr: Optional[Union[str, Pattern[str]]]=None, *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        super().__init__(_ispytest=True)
        msg = 'exceptions must be derived from Warning, not %s'
        if isinstance(expected_warning, tuple):
            for exc in expected_warning:
                if not issubclass(exc, Warning):
                    raise TypeError(msg % type(exc))
            expected_warning_tup = expected_warning
        elif isinstance(expected_warning, type) and issubclass(expected_warning, Warning):
            expected_warning_tup = (expected_warning,)
        else:
            raise TypeError(msg % type(expected_warning))
        self.expected_warning = expected_warning_tup
        self.match_expr = match_expr

    def matches(self, warning: warnings.WarningMessage) -> bool:
        assert self.expected_warning is not None
        return issubclass(warning.category, self.expected_warning) and bool(self.match_expr is None or re.search(self.match_expr, str(warning.message)))

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        __tracebackhide__ = True
        if exc_val is not None and (not isinstance(exc_val, Exception) or isinstance(exc_val, Exit)):
            return

        def found_str() -> str:
            return pformat([record.message for record in self], indent=2)
        try:
            if not any((issubclass(w.category, self.expected_warning) for w in self)):
                fail(f'DID NOT WARN. No warnings of type {self.expected_warning} were emitted.\n Emitted warnings: {found_str()}.')
            elif not any((self.matches(w) for w in self)):
                fail(f'DID NOT WARN. No warnings of type {self.expected_warning} matching the regex were emitted.\n Regex: {self.match_expr}\n Emitted warnings: {found_str()}.')
        finally:
            for w in self:
                if not self.matches(w):
                    warnings.warn_explicit(message=w.message, category=w.category, filename=w.filename, lineno=w.lineno, module=w.__module__, source=w.source)
            for w in self:
                if type(w.message) is not UserWarning:
                    continue
                if not w.message.args:
                    continue
                msg = w.message.args[0]
                if isinstance(msg, str):
                    continue
                raise TypeError(f'Warning must be str or Warning, got {msg!r} (type {type(msg).__name__})')