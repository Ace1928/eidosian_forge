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
def _resolve_one(self, requirement_set: RequirementSet, req_to_install: InstallRequirement) -> List[InstallRequirement]:
    """Prepare a single requirements file.

        :return: A list of additional InstallRequirements to also install.
        """
    if req_to_install.constraint or req_to_install.prepared:
        return []
    req_to_install.prepared = True
    dist = self._get_dist_for(req_to_install)
    _check_dist_requires_python(dist, version_info=self._py_version_info, ignore_requires_python=self.ignore_requires_python)
    more_reqs: List[InstallRequirement] = []

    def add_req(subreq: Requirement, extras_requested: Iterable[str]) -> None:
        sub_install_req = self._make_install_req(str(subreq), req_to_install)
        parent_req_name = req_to_install.name
        to_scan_again, add_to_parent = self._add_requirement_to_set(requirement_set, sub_install_req, parent_req_name=parent_req_name, extras_requested=extras_requested)
        if parent_req_name and add_to_parent:
            self._discovered_dependencies[parent_req_name].append(add_to_parent)
        more_reqs.extend(to_scan_again)
    with indent_log():
        if not requirement_set.has_requirement(req_to_install.name):
            assert req_to_install.user_supplied
            self._add_requirement_to_set(requirement_set, req_to_install, parent_req_name=None)
        if not self.ignore_dependencies:
            if req_to_install.extras:
                logger.debug('Installing extra requirements: %r', ','.join(req_to_install.extras))
            missing_requested = sorted(set(req_to_install.extras) - set(dist.iter_provided_extras()))
            for missing in missing_requested:
                logger.warning("%s %s does not provide the extra '%s'", dist.raw_name, dist.version, missing)
            available_requested = sorted(set(dist.iter_provided_extras()) & set(req_to_install.extras))
            for subreq in dist.iter_dependencies(available_requested):
                add_req(subreq, extras_requested=available_requested)
    return more_reqs