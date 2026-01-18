import sys
from typing import Iterator, Optional
from urllib.parse import ParseResult, urlparse
from .config import ConfigDict, SectionLike
Returns credential sections from the config which match the given URL.