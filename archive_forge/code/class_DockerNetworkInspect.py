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
class DockerNetworkInspect:
    """The results of `docker network inspect` for a single network."""

    def __init__(self, args: CommonConfig, inspection: dict[str, t.Any]) -> None:
        self.args = args
        self.inspection = inspection