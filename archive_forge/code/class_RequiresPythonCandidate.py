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
class RequiresPythonCandidate(Candidate):
    is_installed = False
    source_link = None

    def __init__(self, py_version_info: Optional[Tuple[int, ...]]) -> None:
        if py_version_info is not None:
            version_info = normalize_version_info(py_version_info)
        else:
            version_info = sys.version_info[:3]
        self._version = Version('.'.join((str(c) for c in version_info)))

    def __str__(self) -> str:
        return f'Python {self._version}'

    @property
    def project_name(self) -> NormalizedName:
        return REQUIRES_PYTHON_IDENTIFIER

    @property
    def name(self) -> str:
        return REQUIRES_PYTHON_IDENTIFIER

    @property
    def version(self) -> CandidateVersion:
        return self._version

    def format_for_error(self) -> str:
        return f'Python {self.version}'

    def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
        return ()

    def get_install_requirement(self) -> Optional[InstallRequirement]:
        return None