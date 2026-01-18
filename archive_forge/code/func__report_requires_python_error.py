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
def _report_requires_python_error(self, causes: Sequence['ConflictCause']) -> UnsupportedPythonVersion:
    assert causes, 'Requires-Python error reported with no cause'
    version = self._python_candidate.version
    if len(causes) == 1:
        specifier = str(causes[0].requirement.specifier)
        message = f'Package {causes[0].parent.name!r} requires a different Python: {version} not in {specifier!r}'
        return UnsupportedPythonVersion(message)
    message = f'Packages require a different Python. {version} not in:'
    for cause in causes:
        package = cause.parent.format_for_error()
        specifier = str(cause.requirement.specifier)
        message += f'\n{specifier!r} (required by {package})'
    return UnsupportedPythonVersion(message)