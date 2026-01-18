import collections
import email.message
import functools
import itertools
import json
import logging
import os
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from optparse import Values
from typing import (
from pip._vendor import requests
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import redact_auth_from_url
from pip._internal.vcs import vcs
from .sources import CandidatesFromPage, LinkSource, build_source
def _get_encoding_from_headers(headers: ResponseHeaders) -> Optional[str]:
    """Determine if we have any encoding information in our headers."""
    if headers and 'Content-Type' in headers:
        m = email.message.Message()
        m['content-type'] = headers['Content-Type']
        charset = m.get_param('charset')
        if charset:
            return str(charset)
    return None