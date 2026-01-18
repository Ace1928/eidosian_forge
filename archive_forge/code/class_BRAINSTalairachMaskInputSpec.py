import os
from ...base import (
class BRAINSTalairachMaskInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Input image used to define physical space of resulting mask', exists=True, argstr='--inputVolume %s')
    talairachParameters = File(desc='Name of the Talairach parameter file.', exists=True, argstr='--talairachParameters %s')
    talairachBox = File(desc='Name of the Talairach box file.', exists=True, argstr='--talairachBox %s')
    hemisphereMode = traits.Enum('left', 'right', 'both', desc='Mode for box creation: left, right, both', argstr='--hemisphereMode %s')
    expand = traits.Bool(desc='Expand exterior box to include surface CSF', argstr='--expand ')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Output filename for the resulting binary image', argstr='--outputVolume %s')