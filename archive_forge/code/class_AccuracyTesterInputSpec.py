from ..base import (
import os
class AccuracyTesterInputSpec(CommandLineInputSpec):
    mel_icas = InputMultiPath(Directory(exists=True), copyfile=False, desc='Melodic output directories', argstr='%s', position=3, mandatory=True)
    trained_wts_file = File(desc='trained-weights file', argstr='%s', position=1, mandatory=True)
    output_directory = Directory(desc='Path to folder in which to store the results of the accuracy test.', argstr='%s', position=2, mandatory=True)