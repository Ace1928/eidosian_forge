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
def _set_mapnode_input(self, name, newvalue):
    logger.debug('setting mapnode(%s) input: %s -> %s', str(self), name, str(newvalue))
    if name in self.iterfield:
        setattr(self._inputs, name, newvalue)
    else:
        setattr(self._interface.inputs, name, newvalue)