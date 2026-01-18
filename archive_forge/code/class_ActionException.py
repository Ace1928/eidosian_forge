from __future__ import annotations
from typing import TYPE_CHECKING
from .._exceptions import HCloudException
from ..core import BaseDomain
class ActionException(HCloudException):
    """A generic action exception"""

    def __init__(self, action: Action | BoundAction):
        assert self.__doc__ is not None
        message = self.__doc__
        if action.error is not None and 'message' in action.error:
            message += f': {action.error['message']}'
        super().__init__(message)
        self.message = message
        self.action = action