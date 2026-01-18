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
def _use_static_config(self) -> bool:
    """Use a static config file if available. Returns True if static config is used, otherwise returns False."""
    if os.path.isfile(self.config_static_path):
        display.info('Using existing %s cloud config: %s' % (self.platform, self.config_static_path), verbosity=1)
        self.config_path = self.config_static_path
        static = True
    else:
        static = False
    self.managed = not static
    return static