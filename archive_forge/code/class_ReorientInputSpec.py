from ..utils.filemanip import fname_presuffix
from .base import SimpleInterface, TraitedSpec, BaseInterfaceInputSpec, traits, File
from .. import LooseVersion
class ReorientInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Input image')
    orientation = traits.Enum(_orientations, usedefault=True, desc='Target axis orientation')