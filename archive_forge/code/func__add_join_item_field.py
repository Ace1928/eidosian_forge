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
def _add_join_item_field(self, field, index):
    """Add new join item fields qualified by the given index

        Return the new field name
        """
    name = '%sJ%d' % (field, index + 1)
    trait = self._inputs.trait(field, False, True)
    self._inputs.add_trait(name, trait)
    return name