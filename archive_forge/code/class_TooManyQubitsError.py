import logging
import sys
from types import TracebackType
from typing import Callable, Type
from pyquil.api._logger import logger
class TooManyQubitsError(ApiError):

    def __init__(self, server_status: str):
        explanation = '\nYou requested too many qubits on the QVM.'
        super(TooManyQubitsError, self).__init__(server_status, explanation)