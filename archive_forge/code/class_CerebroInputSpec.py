import os
import re as regex
from ..base import (
class CerebroInputSpec(CommandLineInputSpec):
    inputMRIFile = File(mandatory=True, desc='input 3D MRI volume', argstr='-i %s')
    inputAtlasMRIFile = File(mandatory=True, desc='atlas MRI volume', argstr='--atlas %s')
    inputAtlasLabelFile = File(mandatory=True, desc='atlas labeling', argstr='--atlaslabels %s')
    inputBrainMaskFile = File(desc='brain mask file', argstr='-m %s')
    outputCerebrumMaskFile = File(desc='output cerebrum mask volume. If unspecified, output file name will be auto generated.', argstr='-o %s', genfile=True)
    outputLabelVolumeFile = File(desc='output labeled hemisphere/cerebrum volume. If unspecified, output file name will be auto generated.', argstr='-l %s', genfile=True)
    costFunction = traits.Int(2, usedefault=True, desc='0,1,2', argstr='-c %d')
    useCentroids = traits.Bool(desc='use centroids of data to initialize position', argstr='--centroids')
    outputAffineTransformFile = File(desc='save affine transform to file.', argstr='--air %s', genfile=True)
    outputWarpTransformFile = File(desc='save warp transform to file.', argstr='--warp %s', genfile=True)
    verbosity = traits.Int(desc='verbosity level (0=silent)', argstr='-v %d')
    linearConvergence = traits.Float(desc='linear convergence', argstr='--linconv %f')
    warpLabel = traits.Int(desc='warp order (2,3,4,5,6,7,8)', argstr='--warplevel %d')
    warpConvergence = traits.Float(desc='warp convergence', argstr='--warpconv %f')
    keepTempFiles = traits.Bool(desc="don't remove temporary files", argstr='--keep')
    tempDirectory = traits.Str(desc='specify directory to use for temporary files', argstr='--tempdir %s')
    tempDirectoryBase = traits.Str(desc='create a temporary directory within this directory', argstr='--tempdirbase %s')