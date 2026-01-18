from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
def _call_userfunc_token(self, token):
    try:
        f = getattr(self, token.type)
    except AttributeError:
        return self.__default_token__(token)
    else:
        try:
            return f(token)
        except GrammarError:
            raise
        except Exception as e:
            raise VisitError(token.type, token, e)