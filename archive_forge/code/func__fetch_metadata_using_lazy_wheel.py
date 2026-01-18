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
def _fetch_metadata_using_lazy_wheel(self, link: Link) -> Optional[BaseDistribution]:
    """Fetch metadata using lazy wheel, if possible."""
    if not self.use_lazy_wheel:
        return None
    if link.is_file or not link.is_wheel:
        logger.debug('Lazy wheel is not used as %r does not point to a remote wheel', link)
        return None
    wheel = Wheel(link.filename)
    name = canonicalize_name(wheel.name)
    logger.info('Obtaining dependency information from %s %s', name, wheel.version)
    url = link.url.split('#', 1)[0]
    try:
        return dist_from_wheel_url(name, url, self._session)
    except HTTPRangeRequestUnsupported:
        logger.debug('%s does not support range requests', url)
        return None