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
class CoreHttpError(HttpError):
    """HTTP response as an error."""

    def __init__(self, status: int, remote_message: str, remote_stack_trace: str) -> None:
        super().__init__(status, f'{remote_message}{remote_stack_trace}')
        self.remote_message = remote_message
        self.remote_stack_trace = remote_stack_trace