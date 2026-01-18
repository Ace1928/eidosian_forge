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
def _process_range_request(self, environ: WSGIEnvironment, complete_length: int | None, accept_ranges: bool | str) -> bool:
    """Handle Range Request related headers (RFC7233).  If `Accept-Ranges`
        header is valid, and Range Request is processable, we set the headers
        as described by the RFC, and wrap the underlying response in a
        RangeWrapper.

        Returns ``True`` if Range Request can be fulfilled, ``False`` otherwise.

        :raises: :class:`~werkzeug.exceptions.RequestedRangeNotSatisfiable`
                 if `Range` header could not be parsed or satisfied.

        .. versionchanged:: 2.0
            Returns ``False`` if the length is 0.
        """
    from ..exceptions import RequestedRangeNotSatisfiable
    if not accept_ranges or complete_length is None or complete_length == 0 or (not self._is_range_request_processable(environ)):
        return False
    if accept_ranges is True:
        accept_ranges = 'bytes'
    parsed_range = parse_range_header(environ.get('HTTP_RANGE'))
    if parsed_range is None:
        raise RequestedRangeNotSatisfiable(complete_length)
    range_tuple = parsed_range.range_for_length(complete_length)
    content_range_header = parsed_range.to_content_range_header(complete_length)
    if range_tuple is None or content_range_header is None:
        raise RequestedRangeNotSatisfiable(complete_length)
    content_length = range_tuple[1] - range_tuple[0]
    self.headers['Content-Length'] = str(content_length)
    self.headers['Accept-Ranges'] = accept_ranges
    self.content_range = content_range_header
    self.status_code = 206
    self._wrap_range_response(range_tuple[0], content_length)
    return True