from __future__ import annotations
import abc
import datetime
import os
import re
import tempfile
import time
import typing as t
from ....encoding import (
from ....io import (
from ....util import (
from ....util_common import (
from ....target import (
from ....config import (
from ....ci import (
from ....data import (
from ....docker_util import (
class CloudEnvironment(CloudBase):
    """Base class for cloud environment plugins. Updates integration test environment after delegation."""

    def setup_once(self) -> None:
        """Run setup if it has not already been run."""
        if self.setup_executed:
            return
        self.setup()
        self.setup_executed = True

    def setup(self) -> None:
        """Setup which should be done once per environment instead of once per test target."""

    @abc.abstractmethod
    def get_environment_config(self) -> CloudEnvironmentConfig:
        """Return environment configuration for use in the test environment after delegation."""

    def on_failure(self, target: IntegrationTarget, tries: int) -> None:
        """Callback to run when an integration target fails."""