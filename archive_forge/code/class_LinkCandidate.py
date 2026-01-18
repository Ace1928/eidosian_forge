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
class LinkCandidate(_InstallRequirementBackedCandidate):
    is_editable = False

    def __init__(self, link: Link, template: InstallRequirement, factory: 'Factory', name: Optional[NormalizedName]=None, version: Optional[CandidateVersion]=None) -> None:
        source_link = link
        cache_entry = factory.get_wheel_cache_entry(source_link, name)
        if cache_entry is not None:
            logger.debug('Using cached wheel link: %s', cache_entry.link)
            link = cache_entry.link
        ireq = make_install_req_from_link(link, template)
        assert ireq.link == link
        if ireq.link.is_wheel and (not ireq.link.is_file):
            wheel = Wheel(ireq.link.filename)
            wheel_name = canonicalize_name(wheel.name)
            assert name == wheel_name, f'{name!r} != {wheel_name!r} for wheel'
            if version is not None:
                wheel_version = Version(wheel.version)
                assert version == wheel_version, '{!r} != {!r} for wheel {}'.format(version, wheel_version, name)
        if cache_entry is not None:
            assert ireq.link.is_wheel
            assert ireq.link.is_file
            if cache_entry.persistent and template.link is template.original_link:
                ireq.cached_wheel_source_link = source_link
            if cache_entry.origin is not None:
                ireq.download_info = cache_entry.origin
            else:
                ireq.download_info = direct_url_from_link(source_link, link_is_in_wheel_cache=cache_entry.persistent)
        super().__init__(link=link, source_link=source_link, ireq=ireq, factory=factory, name=name, version=version)

    def _prepare_distribution(self) -> BaseDistribution:
        preparer = self._factory.preparer
        return preparer.prepare_linked_requirement(self._ireq, parallel_builds=True)