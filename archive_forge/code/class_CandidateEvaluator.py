import enum
import functools
import itertools
import logging
import re
from typing import TYPE_CHECKING, FrozenSet, Iterable, List, Optional, Set, Tuple, Union
from pip._vendor.packaging import specifiers
from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import _BaseVersion
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.exceptions import (
from pip._internal.index.collector import LinkCollector, parse_links
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.format_control import FormatControl
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.models.wheel import Wheel
from pip._internal.req import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import build_netloc
from pip._internal.utils.packaging import check_requires_python
from pip._internal.utils.unpacking import SUPPORTED_EXTENSIONS
class CandidateEvaluator:
    """
    Responsible for filtering and sorting candidates for installation based
    on what tags are valid.
    """

    @classmethod
    def create(cls, project_name: str, target_python: Optional[TargetPython]=None, prefer_binary: bool=False, allow_all_prereleases: bool=False, specifier: Optional[specifiers.BaseSpecifier]=None, hashes: Optional[Hashes]=None) -> 'CandidateEvaluator':
        """Create a CandidateEvaluator object.

        :param target_python: The target Python interpreter to use when
            checking compatibility. If None (the default), a TargetPython
            object will be constructed from the running Python.
        :param specifier: An optional object implementing `filter`
            (e.g. `packaging.specifiers.SpecifierSet`) to filter applicable
            versions.
        :param hashes: An optional collection of allowed hashes.
        """
        if target_python is None:
            target_python = TargetPython()
        if specifier is None:
            specifier = specifiers.SpecifierSet()
        supported_tags = target_python.get_sorted_tags()
        return cls(project_name=project_name, supported_tags=supported_tags, specifier=specifier, prefer_binary=prefer_binary, allow_all_prereleases=allow_all_prereleases, hashes=hashes)

    def __init__(self, project_name: str, supported_tags: List[Tag], specifier: specifiers.BaseSpecifier, prefer_binary: bool=False, allow_all_prereleases: bool=False, hashes: Optional[Hashes]=None) -> None:
        """
        :param supported_tags: The PEP 425 tags supported by the target
            Python in order of preference (most preferred first).
        """
        self._allow_all_prereleases = allow_all_prereleases
        self._hashes = hashes
        self._prefer_binary = prefer_binary
        self._project_name = project_name
        self._specifier = specifier
        self._supported_tags = supported_tags
        self._wheel_tag_preferences = {tag: idx for idx, tag in enumerate(supported_tags)}

    def get_applicable_candidates(self, candidates: List[InstallationCandidate]) -> List[InstallationCandidate]:
        """
        Return the applicable candidates from a list of candidates.
        """
        allow_prereleases = self._allow_all_prereleases or None
        specifier = self._specifier
        versions = {str(v) for v in specifier.filter((str(c.version) for c in candidates), prereleases=allow_prereleases)}
        applicable_candidates = [c for c in candidates if str(c.version) in versions]
        filtered_applicable_candidates = filter_unallowed_hashes(candidates=applicable_candidates, hashes=self._hashes, project_name=self._project_name)
        return sorted(filtered_applicable_candidates, key=self._sort_key)

    def _sort_key(self, candidate: InstallationCandidate) -> CandidateSortingKey:
        """
        Function to pass as the `key` argument to a call to sorted() to sort
        InstallationCandidates by preference.

        Returns a tuple such that tuples sorting as greater using Python's
        default comparison operator are more preferred.

        The preference is as follows:

        First and foremost, candidates with allowed (matching) hashes are
        always preferred over candidates without matching hashes. This is
        because e.g. if the only candidate with an allowed hash is yanked,
        we still want to use that candidate.

        Second, excepting hash considerations, candidates that have been
        yanked (in the sense of PEP 592) are always less preferred than
        candidates that haven't been yanked. Then:

        If not finding wheels, they are sorted by version only.
        If finding wheels, then the sort order is by version, then:
          1. existing installs
          2. wheels ordered via Wheel.support_index_min(self._supported_tags)
          3. source archives
        If prefer_binary was set, then all wheels are sorted above sources.

        Note: it was considered to embed this logic into the Link
              comparison operators, but then different sdist links
              with the same version, would have to be considered equal
        """
        valid_tags = self._supported_tags
        support_num = len(valid_tags)
        build_tag: BuildTag = ()
        binary_preference = 0
        link = candidate.link
        if link.is_wheel:
            wheel = Wheel(link.filename)
            try:
                pri = -wheel.find_most_preferred_tag(valid_tags, self._wheel_tag_preferences)
            except ValueError:
                raise UnsupportedWheel(f"{wheel.filename} is not a supported wheel for this platform. It can't be sorted.")
            if self._prefer_binary:
                binary_preference = 1
            if wheel.build_tag is not None:
                match = re.match('^(\\d+)(.*)$', wheel.build_tag)
                assert match is not None, 'guaranteed by filename validation'
                build_tag_groups = match.groups()
                build_tag = (int(build_tag_groups[0]), build_tag_groups[1])
        else:
            pri = -support_num
        has_allowed_hash = int(link.is_hash_allowed(self._hashes))
        yank_value = -1 * int(link.is_yanked)
        return (has_allowed_hash, yank_value, binary_preference, candidate.version, pri, build_tag)

    def sort_best_candidate(self, candidates: List[InstallationCandidate]) -> Optional[InstallationCandidate]:
        """
        Return the best candidate per the instance's sort order, or None if
        no candidate is acceptable.
        """
        if not candidates:
            return None
        best_candidate = max(candidates, key=self._sort_key)
        return best_candidate

    def compute_best_candidate(self, candidates: List[InstallationCandidate]) -> BestCandidateResult:
        """
        Compute and return a `BestCandidateResult` instance.
        """
        applicable_candidates = self.get_applicable_candidates(candidates)
        best_candidate = self.sort_best_candidate(applicable_candidates)
        return BestCandidateResult(candidates, applicable_candidates=applicable_candidates, best_candidate=best_candidate)