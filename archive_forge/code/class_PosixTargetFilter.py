from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
class PosixTargetFilter(TargetFilter[TPosixConfig]):
    """Target filter for POSIX hosts."""

    def filter_targets(self, targets: list[IntegrationTarget], exclude: set[str]) -> None:
        """Filter the list of targets, adding any which this host profile cannot support to the provided exclude list."""
        super().filter_targets(targets, exclude)
        if not self.allow_root and (not self.config.have_root):
            self.skip('needs/root', 'which require --allow-root or running as root', targets, exclude)
        self.skip(f'skip/python{self.config.python.version}', f'which are not supported by Python {self.config.python.version}', targets, exclude)
        self.skip(f'skip/python{self.config.python.major_version}', f'which are not supported by Python {self.config.python.major_version}', targets, exclude)