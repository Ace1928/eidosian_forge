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
def _collate_join_field_inputs(self):
    """
        Collects each override join item field into the interface join
        field input."""
    for field in self.inputs.copyable_trait_names():
        if field in self.joinfield:
            val = self._collate_input_value(field)
            try:
                setattr(self._interface.inputs, field, val)
            except Exception as e:
                raise ValueError('>>JN %s %s %s %s %s: %s' % (self, field, val, self.inputs.copyable_trait_names(), self.joinfield, e))
        elif hasattr(self._interface.inputs, field):
            val = getattr(self._inputs, field)
            if isdefined(val):
                setattr(self._interface.inputs, field, val)
    logger.debug('Collated %d inputs into the %s node join fields', self._next_slot_index, self)