import re
from typing import TYPE_CHECKING, Iterable, Optional, Type, Union, cast
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from w3lib.url import *
from w3lib.url import _safe_chars, _unquotepath  # noqa: F401
from scrapy.utils.python import to_unicode
def _is_posix_path(string: str) -> bool:
    return bool(re.match('\n            ^                   # start with...\n            (\n                \\.              # ...a single dot,\n                (\n                    \\. | [^/\\.]+  # optionally followed by\n                )?                # either a second dot or some characters\n                |\n                ~   # $HOME\n            )?      # optional match of ".", ".." or ".blabla"\n            /       # at least one "/" for a file path,\n            .       # and something after the "/"\n            ', string, flags=re.VERBOSE))