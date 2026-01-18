from nipype.interfaces.base import (
import os
class LandmarksCompareInputSpec(CommandLineInputSpec):
    inputLandmarkFile1 = File(desc='First input landmark file (.fcsv or .wts)', exists=True, argstr='--inputLandmarkFile1 %s')
    inputLandmarkFile2 = File(desc='Second input landmark file (.fcsv or .wts)', exists=True, argstr='--inputLandmarkFile2 %s')
    tolerance = traits.Float(desc='The maximum error (in mm) allowed in each direction of a landmark', argstr='--tolerance %f')