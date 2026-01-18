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
def __makeSimpleAttribute(value: Any, type: Type[T]) -> Attribute[T]:
    if value is None or isinstance(value, type):
        return _ValuedAttribute(value)
    else:
        return _BadAttribute(value, type)