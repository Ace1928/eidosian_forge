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
def _ensure_sequence(self, mutable: bool=False) -> None:
    """This method can be called by methods that need a sequence.  If
        `mutable` is true, it will also ensure that the response sequence
        is a standard Python list.

        .. versionadded:: 0.6
        """
    if self.is_sequence:
        if mutable and (not isinstance(self.response, list)):
            self.response = list(self.response)
        return
    if self.direct_passthrough:
        raise RuntimeError('Attempted implicit sequence conversion but the response object is in direct passthrough mode.')
    if not self.implicit_sequence_conversion:
        raise RuntimeError('The response object required the iterable to be a sequence, but the implicit conversion was disabled. Call make_sequence() yourself.')
    self.make_sequence()