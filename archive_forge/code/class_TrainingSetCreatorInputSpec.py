from ..base import (
import os
class TrainingSetCreatorInputSpec(BaseInterfaceInputSpec):
    mel_icas_in = InputMultiPath(Directory(exists=True), copyfile=False, desc='Melodic output directories', argstr='%s', position=-1)