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
def _create_dynamic_traits(self, basetraits, fields=None, nitems=None):
    """Convert specific fields of a trait to accept multiple inputs"""
    output = DynamicTraitedSpec()
    if fields is None:
        fields = basetraits.copyable_trait_names()
    for name, spec in list(basetraits.items()):
        if name in fields and (nitems is None or nitems > 1):
            logger.debug('adding multipath trait: %s', name)
            if self.nested:
                output.add_trait(name, InputMultiPath(traits.Any()))
            else:
                output.add_trait(name, InputMultiPath(spec.trait_type))
        else:
            output.add_trait(name, traits.Trait(spec))
        setattr(output, name, Undefined)
        value = getattr(basetraits, name)
        if isdefined(value):
            setattr(output, name, value)
        value = getattr(output, name)
    return output