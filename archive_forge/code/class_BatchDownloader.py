import email.message
import logging
import mimetypes
import os
from typing import Iterable, Optional, Tuple
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._internal.cli.progress_bars import get_download_progress_renderer
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.index import PyPI
from pip._internal.models.link import Link
from pip._internal.network.cache import is_from_cache
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
from pip._internal.utils.misc import format_size, redact_auth_from_url, splitext
class BatchDownloader:

    def __init__(self, session: PipSession, progress_bar: str) -> None:
        self._session = session
        self._progress_bar = progress_bar

    def __call__(self, links: Iterable[Link], location: str) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        """Download the files given by links into location."""
        for link in links:
            try:
                resp = _http_get_download(self._session, link)
            except NetworkConnectionError as e:
                assert e.response is not None
                logger.critical('HTTP error %s while getting %s', e.response.status_code, link)
                raise
            filename = _get_http_response_filename(resp, link)
            filepath = os.path.join(location, filename)
            chunks = _prepare_download(resp, link, self._progress_bar)
            with open(filepath, 'wb') as content_file:
                for chunk in chunks:
                    content_file.write(chunk)
            content_type = resp.headers.get('Content-Type', '')
            yield (link, (filepath, content_type))