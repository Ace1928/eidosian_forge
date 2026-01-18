import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class TalairachAVIInputSpec(FSTraitedSpec):
    in_file = File(argstr='--i %s', exists=True, mandatory=True, desc='input volume')
    out_file = File(argstr='--xfm %s', mandatory=True, exists=False, desc='output xfm file')
    atlas = traits.String(argstr='--atlas %s', desc='alternate target atlas (in freesurfer/average dir)')