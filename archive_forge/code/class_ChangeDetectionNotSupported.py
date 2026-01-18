from __future__ import annotations
import abc
import base64
import json
import os
import tempfile
import typing as t
from ..encoding import (
from ..io import (
from ..config import (
from ..util import (
class ChangeDetectionNotSupported(ApplicationError):
    """Exception for cases where change detection is not supported."""