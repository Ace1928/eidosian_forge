import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
def _makeDictOfStringsToClassesAttribute(self, klass: Type[T_gh], value: Dict[str, Union[int, Dict[str, Union[str, int, None]], Dict[str, Union[str, int]]]]) -> Attribute[Dict[str, T_gh]]:
    if isinstance(value, dict) and all((isinstance(key, str) and isinstance(element, dict) for key, element in value.items())):
        return _ValuedAttribute({key: klass(self._requester, self._headers, element, completed=False) for key, element in value.items()})
    else:
        return _BadAttribute(value, {str: dict})