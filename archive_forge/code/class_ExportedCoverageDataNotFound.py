from __future__ import annotations
import collections.abc as c
import os
import json
import typing as t
from ...target import (
from ...io import (
from ...util import (
from ...util_common import (
from ...executor import (
from ...data import (
from ...host_configs import (
from ...provisioning import (
from . import (
class ExportedCoverageDataNotFound(ApplicationError):
    """Exception when no combined coverage data is present yet is required."""

    def __init__(self) -> None:
        super().__init__('Coverage data must be exported before processing with the `--docker` or `--remote` option.\nExport coverage with `ansible-test coverage combine` using the `--export` option.\nThe exported files must be in the directory: %s/' % ResultType.COVERAGE.relative_path)