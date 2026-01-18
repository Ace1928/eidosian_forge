from nipype.interfaces.base import (
import os
class MaskScalarVolumeInputSpec(CommandLineInputSpec):
    InputVolume = File(position=-3, desc='Input volume to be masked', exists=True, argstr='%s')
    MaskVolume = File(position=-2, desc='Label volume containing the mask', exists=True, argstr='%s')
    OutputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output volume: Input Volume masked by label value from Mask Volume', argstr='%s')
    label = traits.Int(desc='Label value in the Mask Volume to use as the mask', argstr='--label %d')
    replace = traits.Int(desc='Value to use for the output volume outside of the mask', argstr='--replace %d')