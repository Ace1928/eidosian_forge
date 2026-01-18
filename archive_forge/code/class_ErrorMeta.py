import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
class ErrorMeta(Exception):
    """Internal testing exception that makes that carries error metadata."""

    def __init__(self, type: Type[Exception], msg: str, *, id: Tuple[Any, ...]=()) -> None:
        super().__init__('If you are a user and see this message during normal operation please file an issue at https://github.com/pytorch/pytorch/issues. If you are a developer and working on the comparison functions, please `raise ErrorMeta().to_error()` for user facing errors.')
        self.type = type
        self.msg = msg
        self.id = id

    def to_error(self, msg: Optional[Union[str, Callable[[str], str]]]=None) -> Exception:
        if not isinstance(msg, str):
            generated_msg = self.msg
            if self.id:
                generated_msg += f'\n\nThe failure occurred for item {''.join((str([item]) for item in self.id))}'
            msg = msg(generated_msg) if callable(msg) else generated_msg
        return self.type(msg)