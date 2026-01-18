import logging
from typing import Iterable, Optional, Set, Tuple
from pip._internal.build_env import BuildEnvironment
from pip._internal.distributions.base import AbstractDistribution
from pip._internal.exceptions import InstallationError
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.subprocess import runner_with_spinner_message
def _get_build_requires_wheel(self) -> Iterable[str]:
    with self.req.build_env:
        runner = runner_with_spinner_message('Getting requirements to build wheel')
        backend = self.req.pep517_backend
        assert backend is not None
        with backend.subprocess_runner(runner):
            return backend.get_requires_for_build_wheel()