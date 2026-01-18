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
class _IndexDirectorySource(LinkSource):
    """``--[extra-]index-url=<path-to-directory>``.

    This is treated like a remote URL; ``candidates_from_page`` contains logic
    for this by appending ``index.html`` to the link.
    """

    def __init__(self, candidates_from_page: CandidatesFromPage, link: Link) -> None:
        self._candidates_from_page = candidates_from_page
        self._link = link

    @property
    def link(self) -> Optional[Link]:
        return self._link

    def page_candidates(self) -> FoundCandidates:
        yield from self._candidates_from_page(self._link)

    def file_links(self) -> FoundLinks:
        return ()