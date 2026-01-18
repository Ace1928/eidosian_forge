from __future__ import annotations
import warnings
from typing import Any, Generator, List, Optional, Sequence
from .datastructures import Headers, MultipleValuesError
from .exceptions import (
from .extensions import ClientExtensionFactory, Extension
from .headers import (
from .http11 import Request, Response
from .protocol import CLIENT, CONNECTING, OPEN, Protocol, State
from .typing import (
from .uri import WebSocketURI
from .utils import accept_key, generate_key
from .legacy.client import *  # isort:skip  # noqa: I001
from .legacy.client import __all__ as legacy__all__
class ClientConnection(ClientProtocol):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn('ClientConnection was renamed to ClientProtocol', DeprecationWarning)
        super().__init__(*args, **kwargs)