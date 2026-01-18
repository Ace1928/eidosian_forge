from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class RigidTask(DTITKRenameMixin, Rigid):
    pass