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
def _fetch_metadata_using_link_data_attr(self, req: InstallRequirement) -> Optional[BaseDistribution]:
    """Fetch metadata from the data-dist-info-metadata attribute, if possible."""
    metadata_link = req.link.metadata_link()
    if metadata_link is None:
        return None
    assert req.req is not None
    logger.verbose('Obtaining dependency information for %s from %s', req.req, metadata_link)
    metadata_file = get_http_url(metadata_link, self._download, hashes=metadata_link.as_hashes())
    with open(metadata_file.path, 'rb') as f:
        metadata_contents = f.read()
    metadata_dist = get_metadata_distribution(metadata_contents, req.link.filename, req.req.name)
    if canonicalize_name(metadata_dist.raw_name) != canonicalize_name(req.req.name):
        raise MetadataInconsistent(req, 'Name', req.req.name, metadata_dist.raw_name)
    return metadata_dist