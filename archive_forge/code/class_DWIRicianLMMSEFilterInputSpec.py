from nipype.interfaces.base import (
import os
class DWIRicianLMMSEFilterInputSpec(CommandLineInputSpec):
    iter = traits.Int(desc='Number of iterations for the noise removal filter.', argstr='--iter %d')
    re = InputMultiPath(traits.Int, desc='Estimation radius.', sep=',', argstr='--re %s')
    rf = InputMultiPath(traits.Int, desc='Filtering radius.', sep=',', argstr='--rf %s')
    mnvf = traits.Int(desc='Minimum number of voxels in kernel used for filtering.', argstr='--mnvf %d')
    mnve = traits.Int(desc='Minimum number of voxels in kernel used for estimation.', argstr='--mnve %d')
    minnstd = traits.Int(desc='Minimum allowed noise standard deviation.', argstr='--minnstd %d')
    maxnstd = traits.Int(desc='Maximum allowed noise standard deviation.', argstr='--maxnstd %d')
    hrf = traits.Float(desc='How many histogram bins per unit interval.', argstr='--hrf %f')
    uav = traits.Bool(desc='Use absolute value in case of negative square.', argstr='--uav ')
    inputVolume = File(position=-2, desc='Input DWI volume.', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output DWI volume.', argstr='%s')
    compressOutput = traits.Bool(desc='Compress the data of the compressed file using gzip', argstr='--compressOutput ')