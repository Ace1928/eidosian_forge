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
def _ensure_api_response(url: str, session: PipSession) -> None:
    """
    Send a HEAD request to the URL, and ensure the response contains a simple
    API Response.

    Raises `_NotHTTP` if the URL is not available for a HEAD request, or
    `_NotAPIContent` if the content type is not a valid content type.
    """
    scheme, netloc, path, query, fragment = urllib.parse.urlsplit(url)
    if scheme not in {'http', 'https'}:
        raise _NotHTTP()
    resp = session.head(url, allow_redirects=True)
    raise_for_status(resp)
    _ensure_api_header(resp)