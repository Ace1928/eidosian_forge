import functools
from typing import Any, Callable, cast, Optional, Type, TYPE_CHECKING, TypeVar, Union
class GitlabHousekeepingError(GitlabOperationError):
    pass