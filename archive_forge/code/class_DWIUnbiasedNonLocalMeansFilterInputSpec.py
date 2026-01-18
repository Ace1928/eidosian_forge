from nipype.interfaces.base import (
import os
class DWIUnbiasedNonLocalMeansFilterInputSpec(CommandLineInputSpec):
    rs = InputMultiPath(traits.Int, desc='The algorithm search for similar voxels in a neighborhood of this size (larger sizes than the default one are extremely slow).', sep=',', argstr='--rs %s')
    rc = InputMultiPath(traits.Int, desc='Similarity between blocks is measured using windows of this size.', sep=',', argstr='--rc %s')
    hp = traits.Float(desc='This parameter is related to noise; the larger the parameter, the more aggressive the filtering. Should be near 1, and only values between 0.8 and 1.2 are allowed', argstr='--hp %f')
    ng = traits.Int(desc='The number of the closest gradients that are used to jointly filter a given gradient direction (a maximum of 5 is allowed).', argstr='--ng %d')
    re = InputMultiPath(traits.Int, desc='A neighborhood of this size is used to compute the statistics for noise estimation.', sep=',', argstr='--re %s')
    inputVolume = File(position=-2, desc='Input DWI volume.', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output DWI volume.', argstr='%s')