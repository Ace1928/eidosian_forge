from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class UrlExtraError(UrlError):
    code = 'url.extra'
    msg_template = 'URL invalid, extra characters found after valid URL: {extra!r}'