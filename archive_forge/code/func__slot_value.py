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
def _slot_value(self, field, index):
    slot_field = '%sJ%d' % (field, index + 1)
    try:
        return getattr(self._inputs, slot_field)
    except AttributeError as e:
        raise AttributeError('The join node %s does not have a slot field %s to hold the %s value at index %d: %s' % (self, slot_field, field, index, e))