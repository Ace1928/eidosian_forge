from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class UrlSchemePermittedError(UrlError):
    code = 'url.scheme'
    msg_template = 'URL scheme not permitted'

    def __init__(self, allowed_schemes: Set[str]):
        super().__init__(allowed_schemes=allowed_schemes)