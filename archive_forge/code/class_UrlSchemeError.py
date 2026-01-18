from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class UrlSchemeError(UrlError):
    code = 'url.scheme'
    msg_template = 'invalid or missing URL scheme'