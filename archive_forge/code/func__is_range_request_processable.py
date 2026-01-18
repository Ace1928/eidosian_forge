from __future__ import annotations
import json
import typing as t
from http import HTTPStatus
from urllib.parse import urljoin
from ..datastructures import Headers
from ..http import remove_entity_headers
from ..sansio.response import Response as _SansIOResponse
from ..urls import _invalid_iri_to_uri
from ..urls import iri_to_uri
from ..utils import cached_property
from ..wsgi import ClosingIterator
from ..wsgi import get_current_url
from werkzeug._internal import _get_environ
from werkzeug.http import generate_etag
from werkzeug.http import http_date
from werkzeug.http import is_resource_modified
from werkzeug.http import parse_etags
from werkzeug.http import parse_range_header
from werkzeug.wsgi import _RangeWrapper
def _is_range_request_processable(self, environ: WSGIEnvironment) -> bool:
    """Return ``True`` if `Range` header is present and if underlying
        resource is considered unchanged when compared with `If-Range` header.
        """
    return ('HTTP_IF_RANGE' not in environ or not is_resource_modified(environ, self.headers.get('etag'), None, self.headers.get('last-modified'), ignore_if_range=False)) and 'HTTP_RANGE' in environ