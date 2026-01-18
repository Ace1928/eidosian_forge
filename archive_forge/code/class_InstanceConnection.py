from __future__ import annotations
import abc
import dataclasses
import json
import os
import re
import stat
import traceback
import uuid
import time
import typing as t
from .http import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .ci import (
from .data import (
class InstanceConnection:
    """Container for remote instance status and connection details."""

    def __init__(self, running: bool, hostname: t.Optional[str]=None, port: t.Optional[int]=None, username: t.Optional[str]=None, password: t.Optional[str]=None, response_json: t.Optional[dict[str, t.Any]]=None) -> None:
        self.running = running
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.response_json = response_json or {}

    def __str__(self):
        if self.password:
            return f'{self.hostname}:{self.port} [{self.username}:{self.password}]'
        return f'{self.hostname}:{self.port} [{self.username}]'