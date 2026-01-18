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
def _log_preparing_link(self, req: InstallRequirement) -> None:
    """Provide context for the requirement being prepared."""
    if req.link.is_file and (not req.is_wheel_from_cache):
        message = 'Processing %s'
        information = str(display_path(req.link.file_path))
    else:
        message = 'Collecting %s'
        information = redact_auth_from_requirement(req.req) if req.req else str(req)
    if req.req and req.comes_from:
        if isinstance(req.comes_from, str):
            comes_from: Optional[str] = req.comes_from
        else:
            comes_from = req.comes_from.from_path()
        if comes_from:
            information += f' (from {comes_from})'
    if (message, information) != self._previous_requirement_header:
        self._previous_requirement_header = (message, information)
        logger.info(message, information)
    if req.is_wheel_from_cache:
        with indent_log():
            logger.info('Using cached %s', req.link.filename)