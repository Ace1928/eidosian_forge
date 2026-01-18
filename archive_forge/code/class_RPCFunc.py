import pickle
from abc import ABC, abstractmethod
from types import LambdaType
from typing import Any, Callable, Dict
from uuid import uuid4
from triad import ParamDict, SerializableRLock, assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path, to_type
class RPCFunc(RPCHandler):
    """RPCHandler wrapping a python function.

    :param func: a python function
    """

    def __init__(self, func: Callable):
        super().__init__()
        assert_or_throw(callable(func), lambda: ValueError(func))
        self._func = func
        if isinstance(func, LambdaType):
            self._uuid = to_uuid('lambda')
        else:
            self._uuid = to_uuid(get_full_type_path(func))

    def __uuid__(self) -> str:
        """If the underlying function is a static function, then the full
        type path of the function determines the uuid, but for a lambda
        function, the uuid is a constant, so it could be overly
        deterministic

        :return: the unique id
        """
        return self._uuid

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)