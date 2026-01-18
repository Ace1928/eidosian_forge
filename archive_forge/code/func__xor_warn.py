import os
from inspect import isclass
from copy import deepcopy
from warnings import warn
from packaging.version import Version
from traits.trait_errors import TraitError
from traits.trait_handlers import TraitDictObject, TraitListObject
from ...utils.filemanip import md5, hash_infile, hash_timestamp
from .traits_extension import (
from ... import config, __version__
def _xor_warn(self, obj, name, old, new):
    """Generates warnings for xor traits"""
    if isdefined(new):
        trait_spec = self.traits()[name]
        for trait_name in trait_spec.xor:
            if trait_name == name:
                continue
            if isdefined(getattr(self, trait_name)):
                self.trait_set(trait_change_notify=False, **{'%s' % name: Undefined})
                msg = 'Input "%s" is mutually exclusive with input "%s", which is already set' % (name, trait_name)
                raise IOError(msg)