import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
class FSTraitedSpecOpenMP(FSTraitedSpec):
    num_threads = traits.Int(desc='allows for specifying more threads')