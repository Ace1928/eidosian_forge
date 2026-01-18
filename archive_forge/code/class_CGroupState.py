from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import pwd
import typing as t
from ..io import (
from ..util import (
from ..config import (
from ..docker_util import (
from ..host_configs import (
from ..cgroup import (
class CGroupState(enum.Enum):
    """The expected state of a cgroup related mount point."""
    HOST = enum.auto()
    PRIVATE = enum.auto()
    SHADOWED = enum.auto()