from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
class tzoffset(tzinfo):

    def __init__(self, secs_west_of_utc: float, name: Union[None, str]=None) -> None:
        self._offset = timedelta(seconds=-secs_west_of_utc)
        self._name = name or 'fixed'

    def __reduce__(self) -> Tuple[Type['tzoffset'], Tuple[float, str]]:
        return (tzoffset, (-self._offset.total_seconds(), self._name))

    def utcoffset(self, dt: Union[datetime, None]) -> timedelta:
        return self._offset

    def tzname(self, dt: Union[datetime, None]) -> str:
        return self._name

    def dst(self, dt: Union[datetime, None]) -> timedelta:
        return ZERO