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
def _iter_found_candidates(self, ireqs: Sequence[InstallRequirement], specifier: SpecifierSet, hashes: Hashes, prefers_installed: bool, incompatible_ids: Set[int]) -> Iterable[Candidate]:
    if not ireqs:
        return ()
    template = ireqs[0]
    assert template.req, 'Candidates found on index must be PEP 508'
    name = canonicalize_name(template.req.name)
    extras: FrozenSet[str] = frozenset()
    for ireq in ireqs:
        assert ireq.req, 'Candidates found on index must be PEP 508'
        specifier &= ireq.req.specifier
        hashes &= ireq.hashes(trust_internet=False)
        extras |= frozenset(ireq.extras)

    def _get_installed_candidate() -> Optional[Candidate]:
        """Get the candidate for the currently-installed version."""
        if self._force_reinstall:
            return None
        try:
            installed_dist = self._installed_dists[name]
        except KeyError:
            return None
        if not specifier.contains(installed_dist.version, prereleases=True):
            return None
        candidate = self._make_candidate_from_dist(dist=installed_dist, extras=extras, template=template)
        if id(candidate) in incompatible_ids:
            return None
        return candidate

    def iter_index_candidate_infos() -> Iterator[IndexCandidateInfo]:
        result = self._finder.find_best_candidate(project_name=name, specifier=specifier, hashes=hashes)
        icans = list(result.iter_applicable())
        all_yanked = all((ican.link.is_yanked for ican in icans))

        def is_pinned(specifier: SpecifierSet) -> bool:
            for sp in specifier:
                if sp.operator == '===':
                    return True
                if sp.operator != '==':
                    continue
                if sp.version.endswith('.*'):
                    continue
                return True
            return False
        pinned = is_pinned(specifier)
        for ican in reversed(icans):
            if not (all_yanked and pinned) and ican.link.is_yanked:
                continue
            func = functools.partial(self._make_candidate_from_link, link=ican.link, extras=extras, template=template, name=name, version=ican.version)
            yield (ican.version, func)
    return FoundCandidates(iter_index_candidate_infos, _get_installed_candidate(), prefers_installed, incompatible_ids)