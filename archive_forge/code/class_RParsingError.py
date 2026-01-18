import abc
import enum
import inspect
import logging
from typing import Tuple
import typing
import warnings
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from _cffi_backend import FFI  # type: ignore
class RParsingError(Exception):

    def __init__(self, msg: str, status: typing.Optional[PARSING_STATUS]=None):
        full_msg = '{msg} - {status}'.format(msg=msg, status=status)
        super().__init__(full_msg)
        self.status = status