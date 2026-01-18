import mimetypes
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.distributions.installed import InstalledDistribution
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_metadata_distribution
from pip._internal.models.direct_url import ArchiveInfo
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.network.download import BatchDownloader, Downloader
from pip._internal.network.lazy_wheel import (
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.direct_url_helpers import (
from pip._internal.utils.hashes import Hashes, MissingHashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import (
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.unpacking import unpack_file
from pip._internal.vcs import vcs
def _complete_partial_requirements(self, partially_downloaded_reqs: Iterable[InstallRequirement], parallel_builds: bool=False) -> None:
    """Download any requirements which were only fetched by metadata."""
    temp_dir = TempDirectory(kind='unpack', globally_managed=True).path
    links_to_fully_download: Dict[Link, InstallRequirement] = {}
    for req in partially_downloaded_reqs:
        assert req.link
        links_to_fully_download[req.link] = req
    batch_download = self._batch_download(links_to_fully_download.keys(), temp_dir)
    for link, (filepath, _) in batch_download:
        logger.debug('Downloading link %s to %s', link, filepath)
        req = links_to_fully_download[link]
        req.local_file_path = filepath
        self._downloaded[req.link.url] = filepath
        if not req.is_wheel:
            req.needs_unpacked_archive(Path(filepath))
    for req in partially_downloaded_reqs:
        self._prepare_linked_requirement(req, parallel_builds)