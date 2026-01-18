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
def _makeListOfStringsAttribute(value: Union[List[List[str]], List[str], List[Union[str, int]]]) -> Attribute:
    return GithubObject.__makeSimpleListAttribute(value, str)