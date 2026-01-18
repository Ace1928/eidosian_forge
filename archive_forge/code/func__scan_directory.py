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
def _scan_directory(self) -> None:
    """Scans directory once and populates both page_candidates
        and project_name_to_urls at the same time
        """
    for entry in os.scandir(self._path):
        url = path_to_url(entry.path)
        if _is_html_file(url):
            self._page_candidates.append(url)
            continue
        try:
            project_filename = parse_wheel_filename(entry.name)[0]
        except (InvalidWheelFilename, InvalidVersion):
            try:
                project_filename = parse_sdist_filename(entry.name)[0]
            except (InvalidSdistFilename, InvalidVersion):
                continue
        self._project_name_to_urls[project_filename].append(url)
    self._scanned_directory = True