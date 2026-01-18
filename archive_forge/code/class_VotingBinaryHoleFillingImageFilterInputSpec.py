from nipype.interfaces.base import (
import os
class VotingBinaryHoleFillingImageFilterInputSpec(CommandLineInputSpec):
    radius = InputMultiPath(traits.Int, desc='The radius of a hole to be filled', sep=',', argstr='--radius %s')
    majorityThreshold = traits.Int(desc='The number of pixels over 50% that will decide whether an OFF pixel will become ON or not. For example, if the neighborhood of a pixel has 124 pixels (excluding itself), the 50% will be 62, and if you set a Majority threshold of 5, that means that the filter will require 67 or more neighbor pixels to be ON in order to switch the current OFF pixel to ON.', argstr='--majorityThreshold %d')
    background = traits.Int(desc='The value associated with the background (not object)', argstr='--background %d')
    foreground = traits.Int(desc='The value associated with the foreground (object)', argstr='--foreground %d')
    inputVolume = File(position=-2, desc='Input volume to be filtered', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output filtered', argstr='%s')