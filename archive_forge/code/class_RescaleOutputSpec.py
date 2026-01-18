from ..utils.filemanip import fname_presuffix
from .base import SimpleInterface, TraitedSpec, BaseInterfaceInputSpec, traits, File
from .. import LooseVersion
class RescaleOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Rescaled image')