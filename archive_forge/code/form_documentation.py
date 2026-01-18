from typing import Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from lxml.html import (
from parsel.selector import create_root_node
from w3lib.html import strip_html5_whitespace
from scrapy.http.request import Request
from scrapy.http.response.text import TextResponse
from scrapy.utils.python import is_listlike, to_bytes
from scrapy.utils.response import get_base_url

    Returns the clickable element specified in clickdata,
    if the latter is given. If not, it returns the first
    clickable element found
    