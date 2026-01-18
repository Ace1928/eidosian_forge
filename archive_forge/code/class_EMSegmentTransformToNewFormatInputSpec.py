from nipype.interfaces.base import (
import os
class EMSegmentTransformToNewFormatInputSpec(CommandLineInputSpec):
    inputMRMLFileName = File(desc='Active MRML scene that contains EMSegment algorithm parameters in the format before 3.6.3 - please include absolute  file name in path.', exists=True, argstr='--inputMRMLFileName %s')
    outputMRMLFileName = traits.Either(traits.Bool, File(), hash_files=False, desc='Write out the MRML scene after transformation to format 3.6.3 has been made. - has to be in the same directory as the input MRML file due to Slicer Core bug  - please include absolute  file name in path ', argstr='--outputMRMLFileName %s')
    templateFlag = traits.Bool(desc='Set to true if the transformed mrml file should be used as template file ', argstr='--templateFlag ')