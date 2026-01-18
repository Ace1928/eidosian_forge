from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
class NetworkIntegrationConfig(IntegrationConfig):
    """Configuration for the network integration command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args, 'network-integration')
        self.testcase: str = args.testcase