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
class _RemoteFileSource(LinkSource):
    """``--find-links=<url>`` or ``--[extra-]index-url=<url>``.

    This returns:

    * ``page_candidates``: Links listed on an HTML file.
    * ``file_candidates``: The non-HTML file.
    """

    def __init__(self, candidates_from_page: CandidatesFromPage, page_validator: PageValidator, link: Link) -> None:
        self._candidates_from_page = candidates_from_page
        self._page_validator = page_validator
        self._link = link

    @property
    def link(self) -> Optional[Link]:
        return self._link

    def page_candidates(self) -> FoundCandidates:
        if not self._page_validator(self._link):
            return
        yield from self._candidates_from_page(self._link)

    def file_links(self) -> FoundLinks:
        yield self._link