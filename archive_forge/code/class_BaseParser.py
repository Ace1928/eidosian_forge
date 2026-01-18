import sys
from abc import ABC
from asyncio import IncompleteReadError, StreamReader, TimeoutError
from typing import List, Optional, Union
from ..exceptions import (
from ..typing import EncodableT
from .encoders import Encoder
from .socket import SERVER_CLOSED_CONNECTION_ERROR, SocketBuffer
class BaseParser(ABC):
    EXCEPTION_CLASSES = {'ERR': {'max number of clients reached': ConnectionError, 'invalid password': AuthenticationError, "wrong number of arguments for 'auth' command": AuthenticationWrongNumberOfArgsError, "wrong number of arguments for 'AUTH' command": AuthenticationWrongNumberOfArgsError, MODULE_LOAD_ERROR: ModuleError, MODULE_EXPORTS_DATA_TYPES_ERROR: ModuleError, NO_SUCH_MODULE_ERROR: ModuleError, MODULE_UNLOAD_NOT_POSSIBLE_ERROR: ModuleError, **NO_AUTH_SET_ERROR}, 'OOM': OutOfMemoryError, 'WRONGPASS': AuthenticationError, 'EXECABORT': ExecAbortError, 'LOADING': BusyLoadingError, 'NOSCRIPT': NoScriptError, 'READONLY': ReadOnlyError, 'NOAUTH': AuthenticationError, 'NOPERM': NoPermissionError}

    @classmethod
    def parse_error(cls, response):
        """Parse an error response"""
        error_code = response.split(' ')[0]
        if error_code in cls.EXCEPTION_CLASSES:
            response = response[len(error_code) + 1:]
            exception_class = cls.EXCEPTION_CLASSES[error_code]
            if isinstance(exception_class, dict):
                exception_class = exception_class.get(response, ResponseError)
            return exception_class(response)
        return ResponseError(response)

    def on_disconnect(self):
        raise NotImplementedError()

    def on_connect(self, connection):
        raise NotImplementedError()