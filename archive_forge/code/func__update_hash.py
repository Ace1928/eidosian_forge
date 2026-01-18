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
def _update_hash(self):
    for outdatedhash in glob(op.join(self.output_dir(), '_0x*.json')):
        os.remove(outdatedhash)
    _save_hashfile(self._hashvalue, self._hashed_inputs)