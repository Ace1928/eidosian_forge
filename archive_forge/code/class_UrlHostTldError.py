from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class UrlHostTldError(UrlError):
    code = 'url.host'
    msg_template = 'URL host invalid, top level domain required'