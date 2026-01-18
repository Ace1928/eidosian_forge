import os
import platform
from functools import wraps
from pathlib import PurePath, PurePosixPath
from typing import Any, NewType, Union
class LogicalPath(str):
    """A string that represents a path relative to an artifact or run.

    The format of the string is always as a POSIX path, e.g. "foo/bar.txt".

    A neat trick is that you can use this class as if it were a PurePosixPath. E.g.:
    ```
    >>> path = LogicalPath("foo/bar.txt")
    >>> path.parts
    ('foo', 'bar.txt')
    >>> path.parent / "baz.txt"
    'foo/baz.txt'
    >>> type(path.relative_to("foo"))
    LogicalPath
    ```
    """

    def __new__(cls, path: StrPath) -> 'LogicalPath':
        if isinstance(path, LogicalPath):
            return super().__new__(cls, path)
        if hasattr(path, 'as_posix'):
            path = PurePosixPath(path.as_posix())
            return super().__new__(cls, str(path))
        if hasattr(path, '__fspath__'):
            path = path.__fspath__()
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if platform.system() == 'Windows':
            path = path.replace('\\', '/')
        path = PurePath(path).as_posix()
        return super().__new__(cls, str(PurePosixPath(path)))

    def to_path(self) -> PurePosixPath:
        """Convert this path to a PurePosixPath."""
        return PurePosixPath(self)

    def __getattr__(self, attr: str) -> Any:
        """Act like a subclass of PurePosixPath for all methods not defined on str."""
        try:
            result = getattr(self.to_path(), attr)
        except AttributeError as e:
            raise AttributeError(f'LogicalPath has no attribute {attr!r}') from e
        if isinstance(result, PurePosixPath):
            return LogicalPath(result)
        if callable(result):

            @wraps(result)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                inner_result = result(*args, **kwargs)
                if isinstance(inner_result, PurePosixPath):
                    return LogicalPath(inner_result)
                return inner_result
            return wrapper
        return result

    def __truediv__(self, other: StrPath) -> 'LogicalPath':
        """Act like a PurePosixPath for the / operator, but return a LogicalPath."""
        return LogicalPath(self.to_path() / LogicalPath(other))