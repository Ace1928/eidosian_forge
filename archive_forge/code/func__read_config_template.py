from __future__ import annotations
import abc
import datetime
import os
import re
import tempfile
import time
import typing as t
from ....encoding import (
from ....io import (
from ....util import (
from ....util_common import (
from ....target import (
from ....config import (
from ....ci import (
from ....data import (
from ....docker_util import (
def _read_config_template(self) -> str:
    """Read and return the configuration template."""
    lines = read_text_file(self.config_template_path).splitlines()
    lines = [line for line in lines if not line.startswith('#')]
    config = '\n'.join(lines).strip() + '\n'
    return config