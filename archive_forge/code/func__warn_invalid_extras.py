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