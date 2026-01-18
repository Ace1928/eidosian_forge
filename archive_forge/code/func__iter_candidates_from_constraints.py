import contextlib
import functools
import logging
from typing import (
from pip._vendor.packaging.requirements import InvalidRequirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.resolvelib import ResolutionImpossible
from pip._internal.cache import CacheEntry, WheelCache
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_default_environment
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
from pip._internal.req.req_install import (
from pip._internal.resolution.base import InstallRequirementProvider
from pip._internal.utils.compatibility_tags import get_supported
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import Candidate, CandidateVersion, Constraint, Requirement
from .candidates import (
from .found_candidates import FoundCandidates, IndexCandidateInfo
from .requirements import (
def _iter_candidates_from_constraints(self, identifier: str, constraint: Constraint, template: InstallRequirement) -> Iterator[Candidate]:
    """Produce explicit candidates from constraints.

        This creates "fake" InstallRequirement objects that are basically clones
        of what "should" be the template, but with original_link set to link.
        """
    for link in constraint.links:
        self._fail_if_link_is_unsupported_wheel(link)
        candidate = self._make_base_candidate_from_link(link, template=install_req_from_link_and_ireq(link, template), name=canonicalize_name(identifier), version=None)
        if candidate:
            yield candidate