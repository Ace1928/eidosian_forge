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
class ContainerNotFoundError(DockerError):
    """The container identified by `identifier` was not found."""

    def __init__(self, identifier: str) -> None:
        super().__init__('The container "%s" was not found.' % identifier)
        self.identifier = identifier