from nipype.interfaces.base import (
import os
class DWIJointRicianLMMSEFilterInputSpec(CommandLineInputSpec):
    re = InputMultiPath(traits.Int, desc='Estimation radius.', sep=',', argstr='--re %s')
    rf = InputMultiPath(traits.Int, desc='Filtering radius.', sep=',', argstr='--rf %s')
    ng = traits.Int(desc='The number of the closest gradients that are used to jointly filter a given gradient direction (0 to use all).', argstr='--ng %d')
    inputVolume = File(position=-2, desc='Input DWI volume.', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output DWI volume.', argstr='%s')
    compressOutput = traits.Bool(desc='Compress the data of the compressed file using gzip', argstr='--compressOutput ')