import asyncio
import enum
import json
import pathlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import (
from traitlets import Any as Any_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable
@lru_cache(maxsize=1)
def _npm_prefix(self, npm: Text):
    try:
        return subprocess.run([npm, 'prefix', '-g'], check=True, capture_output=True).stdout.decode('utf-8').strip()
    except Exception as e:
        self.log.warn(f'Could not determine npm prefix: {e}')