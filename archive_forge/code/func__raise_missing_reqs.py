import logging
from typing import Iterable, Optional, Set, Tuple
from pip._internal.build_env import BuildEnvironment
from pip._internal.distributions.base import AbstractDistribution
from pip._internal.exceptions import InstallationError
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.subprocess import runner_with_spinner_message
def _raise_missing_reqs(self, missing: Set[str]) -> None:
    format_string = 'Some build dependencies for {requirement} are missing: {missing}.'
    error_message = format_string.format(requirement=self.req, missing=', '.join(map(repr, sorted(missing))))
    raise InstallationError(error_message)