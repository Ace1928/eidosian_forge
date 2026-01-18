import logging
import sys
from typing import TYPE_CHECKING, Any, FrozenSet, Iterable, Optional, Tuple, Union, cast
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import Version
from pip._internal.exceptions import (
from pip._internal.metadata import BaseDistribution
from pip._internal.models.link import Link, links_equivalent
from pip._internal.models.wheel import Wheel
from pip._internal.req.constructors import (
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.direct_url_helpers import direct_url_from_link
from pip._internal.utils.misc import normalize_version_info
from .base import Candidate, CandidateVersion, Requirement, format_name
class ExtrasCandidate(Candidate):
    """A candidate that has 'extras', indicating additional dependencies.

    Requirements can be for a project with dependencies, something like
    foo[extra].  The extras don't affect the project/version being installed
    directly, but indicate that we need additional dependencies. We model that
    by having an artificial ExtrasCandidate that wraps the "base" candidate.

    The ExtrasCandidate differs from the base in the following ways:

    1. It has a unique name, of the form foo[extra]. This causes the resolver
       to treat it as a separate node in the dependency graph.
    2. When we're getting the candidate's dependencies,
       a) We specify that we want the extra dependencies as well.
       b) We add a dependency on the base candidate.
          See below for why this is needed.
    3. We return None for the underlying InstallRequirement, as the base
       candidate will provide it, and we don't want to end up with duplicates.

    The dependency on the base candidate is needed so that the resolver can't
    decide that it should recommend foo[extra1] version 1.0 and foo[extra2]
    version 2.0. Having those candidates depend on foo=1.0 and foo=2.0
    respectively forces the resolver to recognise that this is a conflict.
    """

    def __init__(self, base: BaseCandidate, extras: FrozenSet[str], *, comes_from: Optional[InstallRequirement]=None) -> None:
        """
        :param comes_from: the InstallRequirement that led to this candidate if it
            differs from the base's InstallRequirement. This will often be the
            case in the sense that this candidate's requirement has the extras
            while the base's does not. Unlike the InstallRequirement backed
            candidates, this requirement is used solely for reporting purposes,
            it does not do any leg work.
        """
        self.base = base
        self.extras = frozenset((canonicalize_name(e) for e in extras))
        self._unnormalized_extras = extras.difference(self.extras)
        self._comes_from = comes_from if comes_from is not None else self.base._ireq

    def __str__(self) -> str:
        name, rest = str(self.base).split(' ', 1)
        return '{}[{}] {}'.format(name, ','.join(self.extras), rest)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(base={self.base!r}, extras={self.extras!r})'

    def __hash__(self) -> int:
        return hash((self.base, self.extras))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.base == other.base and self.extras == other.extras
        return False

    @property
    def project_name(self) -> NormalizedName:
        return self.base.project_name

    @property
    def name(self) -> str:
        """The normalised name of the project the candidate refers to"""
        return format_name(self.base.project_name, self.extras)

    @property
    def version(self) -> CandidateVersion:
        return self.base.version

    def format_for_error(self) -> str:
        return '{} [{}]'.format(self.base.format_for_error(), ', '.join(sorted(self.extras)))

    @property
    def is_installed(self) -> bool:
        return self.base.is_installed

    @property
    def is_editable(self) -> bool:
        return self.base.is_editable

    @property
    def source_link(self) -> Optional[Link]:
        return self.base.source_link

    def _warn_invalid_extras(self, requested: FrozenSet[str], valid: FrozenSet[str]) -> None:
        """Emit warnings for invalid extras being requested.

        This emits a warning for each requested extra that is not in the
        candidate's ``Provides-Extra`` list.
        """
        invalid_extras_to_warn = frozenset((extra for extra in requested if extra not in valid and extra in self.extras))
        if not invalid_extras_to_warn:
            return
        for extra in sorted(invalid_extras_to_warn):
            logger.warning("%s %s does not provide the extra '%s'", self.base.name, self.version, extra)

    def _calculate_valid_requested_extras(self) -> FrozenSet[str]:
        """Get a list of valid extras requested by this candidate.

        The user (or upstream dependant) may have specified extras that the
        candidate doesn't support. Any unsupported extras are dropped, and each
        cause a warning to be logged here.
        """
        requested_extras = self.extras.union(self._unnormalized_extras)
        valid_extras = frozenset((extra for extra in requested_extras if self.base.dist.is_extra_provided(extra)))
        self._warn_invalid_extras(requested_extras, valid_extras)
        return valid_extras

    def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
        factory = self.base._factory
        yield factory.make_requirement_from_candidate(self.base)
        if not with_requires:
            return
        valid_extras = self._calculate_valid_requested_extras()
        for r in self.base.dist.iter_dependencies(valid_extras):
            yield from factory.make_requirements_from_spec(str(r), self._comes_from, valid_extras)

    def get_install_requirement(self) -> Optional[InstallRequirement]:
        return None