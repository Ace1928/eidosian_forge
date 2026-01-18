import functools
from typing import Any, Callable, cast, Optional, Type, TYPE_CHECKING, TypeVar, Union
class GitlabUnfollowError(GitlabOperationError):
    pass