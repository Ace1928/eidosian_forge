import re
from typing import TYPE_CHECKING, Iterable, Optional, Type, Union, cast
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from w3lib.url import *
from w3lib.url import _safe_chars, _unquotepath  # noqa: F401
from scrapy.utils.python import to_unicode
def _is_windows_path(string: str) -> bool:
    return bool(re.match('\n            ^\n            (\n                [a-z]:\\\\\n                | \\\\\\\\\n            )\n            ', string, flags=re.IGNORECASE | re.VERBOSE))