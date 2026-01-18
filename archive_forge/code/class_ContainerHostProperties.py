from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
@dataclasses.dataclass(frozen=True)
class ContainerHostProperties:
    """Container host properties detected at run time."""
    audit_code: str
    max_open_files: int
    loginuid: t.Optional[int]
    cgroup_v1: SystemdControlGroupV1Status
    cgroup_v2: bool