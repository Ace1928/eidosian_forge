import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
class _BadAttribute(Attribute):

    def __init__(self, value: Any, expectedType: Any, exception: Optional[Exception]=None):
        self.__value = value
        self.__expectedType = expectedType
        self.__exception = exception

    @property
    def value(self) -> Any:
        raise BadAttributeException(self.__value, self.__expectedType, self.__exception)