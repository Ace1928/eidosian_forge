from nipype.interfaces.base import (
import os
class CastScalarVolumeInputSpec(CommandLineInputSpec):
    InputVolume = File(position=-2, desc='Input volume, the volume to cast.', exists=True, argstr='%s')
    OutputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output volume, cast to the new type.', argstr='%s')
    type = traits.Enum('Char', 'UnsignedChar', 'Short', 'UnsignedShort', 'Int', 'UnsignedInt', 'Float', 'Double', desc='Type for the new output volume.', argstr='--type %s')