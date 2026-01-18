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
def _default_stub_value(source_paths: list[str]) -> dict[str, dict[int, int]]:
    cmd = ['pwsh', os.path.join(ANSIBLE_TEST_TOOLS_ROOT, 'coverage_stub.ps1')]
    cmd.extend(source_paths)
    stubs = json.loads(raw_command(cmd, capture=True)[0])
    return dict(((d['Path'], dict(((line, 0) for line in d['Lines']))) for d in stubs))