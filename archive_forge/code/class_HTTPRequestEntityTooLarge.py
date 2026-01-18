import warnings
from typing import Any, Dict, Iterable, List, Optional, Set  # noqa
from yarl import URL
from .typedefs import LooseHeaders, StrOrURL
from .web_response import Response
class HTTPRequestEntityTooLarge(HTTPClientError):
    status_code = 413

    def __init__(self, max_size: float, actual_size: float, **kwargs: Any) -> None:
        kwargs.setdefault('text', 'Maximum request body size {} exceeded, actual body size {}'.format(max_size, actual_size))
        super().__init__(**kwargs)