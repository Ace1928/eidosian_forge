import functools
from typing import Any, Callable, cast, Optional, Type, TYPE_CHECKING, TypeVar, Union
class GitlabUserRejectError(GitlabOperationError):
    pass