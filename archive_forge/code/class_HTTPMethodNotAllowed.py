import warnings
from typing import Any, Dict, Iterable, List, Optional, Set  # noqa
from yarl import URL
from .typedefs import LooseHeaders, StrOrURL
from .web_response import Response
class HTTPMethodNotAllowed(HTTPClientError):
    status_code = 405

    def __init__(self, method: str, allowed_methods: Iterable[str], *, headers: Optional[LooseHeaders]=None, reason: Optional[str]=None, body: Any=None, text: Optional[str]=None, content_type: Optional[str]=None) -> None:
        allow = ','.join(sorted(allowed_methods))
        super().__init__(headers=headers, reason=reason, body=body, text=text, content_type=content_type)
        self.headers['Allow'] = allow
        self.allowed_methods: Set[str] = set(allowed_methods)
        self.method = method.upper()