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
def _get_index_content(link: Link, *, session: PipSession) -> Optional['IndexContent']:
    url = link.url.split('#', 1)[0]
    vcs_scheme = _match_vcs_scheme(url)
    if vcs_scheme:
        logger.warning('Cannot look at %s URL %s because it does not support lookup as web pages.', vcs_scheme, link)
        return None
    scheme, _, path, _, _, _ = urllib.parse.urlparse(url)
    if scheme == 'file' and os.path.isdir(urllib.request.url2pathname(path)):
        if not url.endswith('/'):
            url += '/'
        url = urllib.parse.urljoin(url, 'index.html')
        logger.debug(' file: URL is directory, getting %s', url)
    try:
        resp = _get_simple_response(url, session=session)
    except _NotHTTP:
        logger.warning('Skipping page %s because it looks like an archive, and cannot be checked by a HTTP HEAD request.', link)
    except _NotAPIContent as exc:
        logger.warning('Skipping page %s because the %s request got Content-Type: %s. The only supported Content-Types are application/vnd.pypi.simple.v1+json, application/vnd.pypi.simple.v1+html, and text/html', link, exc.request_desc, exc.content_type)
    except NetworkConnectionError as exc:
        _handle_get_simple_fail(link, exc)
    except RetryError as exc:
        _handle_get_simple_fail(link, exc)
    except SSLError as exc:
        reason = 'There was a problem confirming the ssl certificate: '
        reason += str(exc)
        _handle_get_simple_fail(link, reason, meth=logger.info)
    except requests.ConnectionError as exc:
        _handle_get_simple_fail(link, f'connection error: {exc}')
    except requests.Timeout:
        _handle_get_simple_fail(link, 'timed out')
    else:
        return _make_index_content(resp, cache_link_parsing=link.cache_link_parsing)
    return None