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
class SanityConfig(TestConfig):
    """Configuration for the sanity command."""

    def __init__(self, args: t.Any) -> None:
        super().__init__(args, 'sanity')
        self.test: list[str] = args.test
        self.skip_test: list[str] = args.skip_test
        self.list_tests: bool = args.list_tests
        self.allow_disabled: bool = args.allow_disabled
        self.enable_optional_errors: bool = args.enable_optional_errors
        self.prime_venvs: bool = args.prime_venvs
        self.display_stderr = self.lint or self.list_tests