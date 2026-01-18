from __future__ import annotations
import typing as T
from .baseobjects import MesonInterpreterObject
class Disabler(MesonInterpreterObject):

    def method_call(self, method_name: str, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> TYPE_var:
        if method_name == 'found':
            return False
        return Disabler()