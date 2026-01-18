from ..base import (
import os
class ClassifierInputSpec(CommandLineInputSpec):
    mel_ica = Directory(exists=True, copyfile=False, desc='Melodic output directory or directories', argstr='%s', position=1)
    trained_wts_file = File(exists=True, desc='trained-weights file', argstr='%s', position=2, mandatory=True, copyfile=False)
    thresh = traits.Int(argstr='%d', desc='Threshold for cleanup.', position=-1, mandatory=True)
    artifacts_list_file = File(desc='Text file listing which ICs are artifacts; can be the output from classification or can be created manually')