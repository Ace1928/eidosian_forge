import logging
import sys
from types import TracebackType
from typing import Callable, Type
from pyquil.api._logger import logger
class ApiError(RuntimeError):

    def __init__(self, server_status: str, explanation: str):
        super(ApiError, self).__init__(self, server_status)
        self.server_status = server_status
        self.explanation = explanation

    def __repr__(self) -> str:
        return repr(str(self))

    def __str__(self) -> str:
        return '{}\n{}'.format(self.server_status, self.explanation)