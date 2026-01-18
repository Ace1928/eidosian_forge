import warnings
from typing import Any, Dict, Iterable, List, Optional, Set  # noqa
from yarl import URL
from .typedefs import LooseHeaders, StrOrURL
from .web_response import Response
class HTTPNoContent(HTTPSuccessful):
    status_code = 204
    empty_body = True