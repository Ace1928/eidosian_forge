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
def _override_join_traits(self, basetraits, fields):
    """Convert the given join fields to accept an input that
        is a list item rather than a list. Non-join fields
        delegate to the interface traits.

        Return the override DynamicTraitedSpec
        """
    dyntraits = DynamicTraitedSpec()
    if fields is None:
        fields = basetraits.copyable_trait_names()
    else:
        for field in fields:
            if not basetraits.trait(field):
                raise ValueError('The JoinNode %s does not have a field named %s' % (self.name, field))
    for name, trait in list(basetraits.items()):
        if name in fields and len(trait.inner_traits) == 1:
            item_trait = trait.inner_traits[0]
            dyntraits.add_trait(name, item_trait)
            setattr(dyntraits, name, Undefined)
            logger.debug('Converted the join node %s field %s trait type from %s to %s', self, name, trait.trait_type.info(), item_trait.info())
        else:
            dyntraits.add_trait(name, traits.Any)
            setattr(dyntraits, name, Undefined)
    return dyntraits