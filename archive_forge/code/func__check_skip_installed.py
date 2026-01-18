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
def _check_skip_installed(self, req_to_install: InstallRequirement) -> Optional[str]:
    """Check if req_to_install should be skipped.

        This will check if the req is installed, and whether we should upgrade
        or reinstall it, taking into account all the relevant user options.

        After calling this req_to_install will only have satisfied_by set to
        None if the req_to_install is to be upgraded/reinstalled etc. Any
        other value will be a dist recording the current thing installed that
        satisfies the requirement.

        Note that for vcs urls and the like we can't assess skipping in this
        routine - we simply identify that we need to pull the thing down,
        then later on it is pulled down and introspected to assess upgrade/
        reinstalls etc.

        :return: A text reason for why it was skipped, or None.
        """
    if self.ignore_installed:
        return None
    req_to_install.check_if_exists(self.use_user_site)
    if not req_to_install.satisfied_by:
        return None
    if self.force_reinstall:
        self._set_req_to_reinstall(req_to_install)
        return None
    if not self._is_upgrade_allowed(req_to_install):
        if self.upgrade_strategy == 'only-if-needed':
            return 'already satisfied, skipping upgrade'
        return 'already satisfied'
    if not req_to_install.link:
        try:
            self.finder.find_requirement(req_to_install, upgrade=True)
        except BestVersionAlreadyInstalled:
            return 'already up-to-date'
        except DistributionNotFound:
            pass
    self._set_req_to_reinstall(req_to_install)
    return None