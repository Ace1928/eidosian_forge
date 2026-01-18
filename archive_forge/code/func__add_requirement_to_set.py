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
def _add_requirement_to_set(self, requirement_set: RequirementSet, install_req: InstallRequirement, parent_req_name: Optional[str]=None, extras_requested: Optional[Iterable[str]]=None) -> Tuple[List[InstallRequirement], Optional[InstallRequirement]]:
    """Add install_req as a requirement to install.

        :param parent_req_name: The name of the requirement that needed this
            added. The name is used because when multiple unnamed requirements
            resolve to the same name, we could otherwise end up with dependency
            links that point outside the Requirements set. parent_req must
            already be added. Note that None implies that this is a user
            supplied requirement, vs an inferred one.
        :param extras_requested: an iterable of extras used to evaluate the
            environment markers.
        :return: Additional requirements to scan. That is either [] if
            the requirement is not applicable, or [install_req] if the
            requirement is applicable and has just been added.
        """
    if not install_req.match_markers(extras_requested):
        logger.info("Ignoring %s: markers '%s' don't match your environment", install_req.name, install_req.markers)
        return ([], None)
    if install_req.link and install_req.link.is_wheel:
        wheel = Wheel(install_req.link.filename)
        tags = compatibility_tags.get_supported()
        if requirement_set.check_supported_wheels and (not wheel.supported(tags)):
            raise InstallationError(f'{wheel.filename} is not a supported wheel on this platform.')
    assert not install_req.user_supplied or parent_req_name is None, "a user supplied req shouldn't have a parent"
    if not install_req.name:
        requirement_set.add_unnamed_requirement(install_req)
        return ([install_req], None)
    try:
        existing_req: Optional[InstallRequirement] = requirement_set.get_requirement(install_req.name)
    except KeyError:
        existing_req = None
    has_conflicting_requirement = parent_req_name is None and existing_req and (not existing_req.constraint) and (existing_req.extras == install_req.extras) and existing_req.req and install_req.req and (existing_req.req.specifier != install_req.req.specifier)
    if has_conflicting_requirement:
        raise InstallationError('Double requirement given: {} (already in {}, name={!r})'.format(install_req, existing_req, install_req.name))
    if not existing_req:
        requirement_set.add_named_requirement(install_req)
        return ([install_req], install_req)
    if install_req.constraint or not existing_req.constraint:
        return ([], existing_req)
    does_not_satisfy_constraint = install_req.link and (not (existing_req.link and install_req.link.path == existing_req.link.path))
    if does_not_satisfy_constraint:
        raise InstallationError(f"Could not satisfy constraints for '{install_req.name}': installation from path or url cannot be constrained to a version")
    existing_req.constraint = False
    if install_req.user_supplied:
        existing_req.user_supplied = True
    existing_req.extras = tuple(sorted(set(existing_req.extras) | set(install_req.extras)))
    logger.debug('Setting %s extras to: %s', existing_req, existing_req.extras)
    return ([existing_req], existing_req)