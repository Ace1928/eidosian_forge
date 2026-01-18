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
def _find_requirement_link(self, req: InstallRequirement) -> Optional[Link]:
    upgrade = self._is_upgrade_allowed(req)
    best_candidate = self.finder.find_requirement(req, upgrade)
    if not best_candidate:
        return None
    link = best_candidate.link
    if link.is_yanked:
        reason = link.yanked_reason or '<none given>'
        msg = f'The candidate selected for download or install is a yanked version: {best_candidate}\nReason for being yanked: {reason}'
        logger.warning(msg)
    return link