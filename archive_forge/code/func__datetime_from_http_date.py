import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
def _datetime_from_http_date(value: str) -> datetime:
    """
    Convert an HTTP date to a datetime object.
    Raises ValueError for invalid dates.
    """
    dt = email.utils.parsedate_to_datetime(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt