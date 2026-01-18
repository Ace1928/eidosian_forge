import logging
import sys
from types import TracebackType
from typing import Callable, Type
from pyquil.api._logger import logger
class UnknownApiError(ApiError):

    def __init__(self, server_status: str):
        explanation = '\nThe server has failed to return a proper response. Please describe the problem\nand copy the above message into a GitHub issue at:\n    https://github.com/rigetti/pyquil/issues'
        super(UnknownApiError, self).__init__(server_status, explanation)