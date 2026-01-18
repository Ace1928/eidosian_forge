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
class _InstallRequirementBackedCandidate(Candidate):
    """A candidate backed by an ``InstallRequirement``.

    This represents a package request with the target not being already
    in the environment, and needs to be fetched and installed. The backing
    ``InstallRequirement`` is responsible for most of the leg work; this
    class exposes appropriate information to the resolver.

    :param link: The link passed to the ``InstallRequirement``. The backing
        ``InstallRequirement`` will use this link to fetch the distribution.
    :param source_link: The link this candidate "originates" from. This is
        different from ``link`` when the link is found in the wheel cache.
        ``link`` would point to the wheel cache, while this points to the
        found remote link (e.g. from pypi.org).
    """
    dist: BaseDistribution
    is_installed = False

    def __init__(self, link: Link, source_link: Link, ireq: InstallRequirement, factory: 'Factory', name: Optional[NormalizedName]=None, version: Optional[CandidateVersion]=None) -> None:
        self._link = link
        self._source_link = source_link
        self._factory = factory
        self._ireq = ireq
        self._name = name
        self._version = version
        self.dist = self._prepare()

    def __str__(self) -> str:
        return f'{self.name} {self.version}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self._link)!r})'

    def __hash__(self) -> int:
        return hash((self.__class__, self._link))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return links_equivalent(self._link, other._link)
        return False

    @property
    def source_link(self) -> Optional[Link]:
        return self._source_link

    @property
    def project_name(self) -> NormalizedName:
        """The normalised name of the project the candidate refers to"""
        if self._name is None:
            self._name = self.dist.canonical_name
        return self._name

    @property
    def name(self) -> str:
        return self.project_name

    @property
    def version(self) -> CandidateVersion:
        if self._version is None:
            self._version = self.dist.version
        return self._version

    def format_for_error(self) -> str:
        return '{} {} (from {})'.format(self.name, self.version, self._link.file_path if self._link.is_file else self._link)

    def _prepare_distribution(self) -> BaseDistribution:
        raise NotImplementedError('Override in subclass')

    def _check_metadata_consistency(self, dist: BaseDistribution) -> None:
        """Check for consistency of project name and version of dist."""
        if self._name is not None and self._name != dist.canonical_name:
            raise MetadataInconsistent(self._ireq, 'name', self._name, dist.canonical_name)
        if self._version is not None and self._version != dist.version:
            raise MetadataInconsistent(self._ireq, 'version', str(self._version), str(dist.version))

    def _prepare(self) -> BaseDistribution:
        try:
            dist = self._prepare_distribution()
        except HashError as e:
            e.req = self._ireq
            raise
        except InstallationSubprocessError as exc:
            exc.context = 'See above for output.'
            raise
        self._check_metadata_consistency(dist)
        return dist

    def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
        requires = self.dist.iter_dependencies() if with_requires else ()
        for r in requires:
            yield from self._factory.make_requirements_from_spec(str(r), self._ireq)
        yield self._factory.make_requires_python_requirement(self.dist.requires_python)

    def get_install_requirement(self) -> Optional[InstallRequirement]:
        return self._ireq