import logging
import sys
from collections import defaultdict
from itertools import chain
from typing import DefaultDict, Iterable, List, Optional, Set, Tuple
from pip._vendor.packaging import specifiers
from pip._vendor.packaging.requirements import Requirement
from pip._internal.cache import WheelCache
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.req_install import (
from pip._internal.req.req_set import RequirementSet
from pip._internal.resolution.base import BaseResolver, InstallRequirementProvider
from pip._internal.utils import compatibility_tags
from pip._internal.utils.compatibility_tags import get_supported
from pip._internal.utils.direct_url_helpers import direct_url_from_link
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import normalize_version_info
from pip._internal.utils.packaging import check_requires_python
def _populate_link(self, req: InstallRequirement) -> None:
    """Ensure that if a link can be found for this, that it is found.

        Note that req.link may still be None - if the requirement is already
        installed and not needed to be upgraded based on the return value of
        _is_upgrade_allowed().

        If preparer.require_hashes is True, don't use the wheel cache, because
        cached wheels, always built locally, have different hashes than the
        files downloaded from the index server and thus throw false hash
        mismatches. Furthermore, cached wheels at present have undeterministic
        contents due to file modification times.
        """
    if req.link is None:
        req.link = self._find_requirement_link(req)
    if self.wheel_cache is None or self.preparer.require_hashes:
        return
    cache_entry = self.wheel_cache.get_cache_entry(link=req.link, package_name=req.name, supported_tags=get_supported())
    if cache_entry is not None:
        logger.debug('Using cached wheel link: %s', cache_entry.link)
        if req.link is req.original_link and cache_entry.persistent:
            req.cached_wheel_source_link = req.link
        if cache_entry.origin is not None:
            req.download_info = cache_entry.origin
        else:
            req.download_info = direct_url_from_link(req.link, link_is_in_wheel_cache=cache_entry.persistent)
        req.link = cache_entry.link