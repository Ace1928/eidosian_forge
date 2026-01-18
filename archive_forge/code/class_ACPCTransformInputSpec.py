from nipype.interfaces.base import (
import os
class ACPCTransformInputSpec(CommandLineInputSpec):
    acpc = InputMultiPath(traits.List(traits.Float(), minlen=3, maxlen=3), desc='ACPC line, two fiducial points, one at the anterior commissure and one at the posterior commissure.', argstr='--acpc %s...')
    midline = InputMultiPath(traits.List(traits.Float(), minlen=3, maxlen=3), desc='The midline is a series of points defining the division between the hemispheres of the brain (the mid sagittal plane).', argstr='--midline %s...')
    outputTransform = traits.Either(traits.Bool, File(), hash_files=False, desc='A transform filled in from the ACPC and Midline registration calculation', argstr='--outputTransform %s')
    debugSwitch = traits.Bool(desc='Click if wish to see debugging output', argstr='--debugSwitch ')