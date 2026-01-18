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
def _generate_handlers(self):
    """Find all traits with the 'xor' metadata and attach an event
        handler to them.
        """
    has_xor = dict(xor=lambda t: t is not None)
    xors = self.trait_names(**has_xor)
    for elem in xors:
        self.on_trait_change(self._xor_warn, elem)
    has_deprecation = dict(deprecated=lambda t: t is not None)
    deprecated = self.trait_names(**has_deprecation)
    for elem in deprecated:
        self.on_trait_change(self._deprecated_warn, elem)