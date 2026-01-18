import logging
from typing import Iterable, Optional, Set, Tuple
from pip._internal.build_env import BuildEnvironment
from pip._internal.distributions.base import AbstractDistribution
from pip._internal.exceptions import InstallationError
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.subprocess import runner_with_spinner_message
def _prepare_build_backend(self, finder: PackageFinder) -> None:
    pyproject_requires = self.req.pyproject_requires
    assert pyproject_requires is not None
    self.req.build_env = BuildEnvironment()
    self.req.build_env.install_requirements(finder, pyproject_requires, 'overlay', kind='build dependencies')
    conflicting, missing = self.req.build_env.check_requirements(self.req.requirements_to_check)
    if conflicting:
        self._raise_conflicts('PEP 517/518 supported requirements', conflicting)
    if missing:
        logger.warning('Missing build requirements in pyproject.toml for %s.', self.req)
        logger.warning('The project does not specify a build backend, and pip cannot fall back to setuptools without %s.', ' and '.join(map(repr, sorted(missing))))