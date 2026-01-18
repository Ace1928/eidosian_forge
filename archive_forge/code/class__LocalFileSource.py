import logging
import mimetypes
import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from pip._vendor.packaging.utils import (
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.link import Link
from pip._internal.utils.urls import path_to_url, url_to_path
from pip._internal.vcs import is_url
class _LocalFileSource(LinkSource):
    """``--find-links=<path-or-url>`` or ``--[extra-]index-url=<path-or-url>``.

    If a URL is supplied, it must be a ``file:`` URL. If a path is supplied to
    the option, it is converted to a URL first. This returns:

    * ``page_candidates``: Links listed on an HTML file.
    * ``file_candidates``: The non-HTML file.
    """

    def __init__(self, candidates_from_page: CandidatesFromPage, link: Link) -> None:
        self._candidates_from_page = candidates_from_page
        self._link = link

    @property
    def link(self) -> Optional[Link]:
        return self._link

    def page_candidates(self) -> FoundCandidates:
        if not _is_html_file(self._link.url):
            return
        yield from self._candidates_from_page(self._link)

    def file_links(self) -> FoundLinks:
        if _is_html_file(self._link.url):
            return
        yield self._link