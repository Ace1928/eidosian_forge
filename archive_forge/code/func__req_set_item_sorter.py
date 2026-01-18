import contextlib
import functools
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, cast
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.resolvelib import BaseReporter, ResolutionImpossible
from pip._vendor.resolvelib import Resolver as RLResolver
from pip._vendor.resolvelib.structs import DirectedGraph
from pip._internal.cache import WheelCache
from pip._internal.index.package_finder import PackageFinder
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import install_req_extend_extras
from pip._internal.req.req_install import InstallRequirement
from pip._internal.req.req_set import RequirementSet
from pip._internal.resolution.base import BaseResolver, InstallRequirementProvider
from pip._internal.resolution.resolvelib.provider import PipProvider
from pip._internal.resolution.resolvelib.reporter import (
from pip._internal.utils.packaging import get_requirement
from .base import Candidate, Requirement
from .factory import Factory
def _req_set_item_sorter(item: Tuple[str, InstallRequirement], weights: Dict[Optional[str], int]) -> Tuple[int, str]:
    """Key function used to sort install requirements for installation.

    Based on the "weight" mapping calculated in ``get_installation_order()``.
    The canonical package name is returned as the second member as a tie-
    breaker to ensure the result is predictable, which is useful in tests.
    """
    name = canonicalize_name(item[0])
    return (weights[name], name)