from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
def _run_command(self, execute, copyfiles=True):
    """Collates the join inputs prior to delegating to the superclass."""
    self._collate_join_field_inputs()
    return super(JoinNode, self)._run_command(execute, copyfiles)