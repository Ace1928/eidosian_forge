from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
class TestSuccess(TestResult):
    """Test success."""

    def write_junit(self, args: TestConfig) -> None:
        """Write results to a junit XML file."""
        test_case = junit_xml.TestCase(classname=self.command, name=self.name)
        self.save_junit(args, test_case)