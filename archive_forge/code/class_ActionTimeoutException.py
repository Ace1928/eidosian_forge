from __future__ import annotations
from typing import TYPE_CHECKING
from .._exceptions import HCloudException
from ..core import BaseDomain
class ActionTimeoutException(ActionException):
    """The pending action timed out"""