from __future__ import annotations
import itertools
import typing
import warnings
import weakref
def _prepare_user_args(self, weak_args: Iterable[typing.Any]=(), user_args: Iterable[typing.Any]=(), callback: Callable[..., typing.Any] | None=None) -> tuple[Collection[weakref.ReferenceType], Collection[typing.Any]]:
    w_args = tuple((weakref.ref(w_arg, callback) for w_arg in weak_args))
    args = tuple(user_args) or ()
    return (w_args, args)