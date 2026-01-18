import os
from ...base import (
class HammerAttributeCreatorInputSpec(CommandLineInputSpec):
    Scale = traits.Int(desc='Determine Scale of Ball', argstr='--Scale %d')
    Strength = traits.Float(desc='Determine Strength of Edges', argstr='--Strength %f')
    inputGMVolume = File(desc='Required: input grey matter posterior image', exists=True, argstr='--inputGMVolume %s')
    inputWMVolume = File(desc='Required: input white matter posterior image', exists=True, argstr='--inputWMVolume %s')
    inputCSFVolume = File(desc='Required: input CSF posterior image', exists=True, argstr='--inputCSFVolume %s')
    outputVolumeBase = traits.Str(desc='Required: output image base name to be appended for each feature vector.', argstr='--outputVolumeBase %s')