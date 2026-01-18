from __future__ import annotations
import os
import platform
import random
import re
import typing as t
from ..config import (
from ..io import (
from ..git import (
from ..util import (
from . import (
@staticmethod
def _get_aci_key_path() -> str:
    path = os.path.expanduser('~/.ansible-core-ci.key')
    return path