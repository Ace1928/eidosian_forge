import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
@staticmethod
def __makeTransformedAttribute(value: T, type: Type[T], transform: Callable[[T], K]) -> Attribute[K]:
    if value is None:
        return _ValuedAttribute(None)
    elif isinstance(value, type):
        try:
            return _ValuedAttribute(transform(value))
        except Exception as e:
            return _BadAttribute(value, type, e)
    else:
        return _BadAttribute(value, type)