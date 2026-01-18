import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class CheckTalairachAlignmentInputSpec(FSTraitedSpec):
    in_file = File(argstr='-xfm %s', xor=['subject'], exists=True, mandatory=True, position=-1, desc='specify the talairach.xfm file to check')
    subject = traits.String(argstr='-subj %s', xor=['in_file'], mandatory=True, position=-1, desc="specify subject's name")
    threshold = traits.Float(default_value=0.01, usedefault=True, argstr='-T %.3f', desc='Talairach transforms for subjects with p-values <= T ' + 'are considered as very unlikely default=0.010')