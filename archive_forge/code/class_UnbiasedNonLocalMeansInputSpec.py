import os
from ...base import (
class UnbiasedNonLocalMeansInputSpec(CommandLineInputSpec):
    sigma = traits.Float(desc='The root power of noise (sigma) in the complex Gaussian process the Rician comes from. If it is underestimated, the algorithm fails to remove the noise. If it is overestimated, over-blurring is likely to occur.', argstr='--sigma %f')
    rs = InputMultiPath(traits.Int, desc='The algorithm search for similar voxels in a neighborhood of this radius (radii larger than 5,5,5 are very slow, and the results can be only marginally better. Small radii may fail to effectively remove the noise).', sep=',', argstr='--rs %s')
    rc = InputMultiPath(traits.Int, desc='Similarity between blocks is computed as the difference between mean values and gradients. These parameters are computed fitting a hyperplane with LS inside a neighborhood of this size', sep=',', argstr='--rc %s')
    hp = traits.Float(desc='This parameter is related to noise; the larger the parameter, the more aggressive the filtering. Should be near 1, and only values between 0.8 and 1.2 are allowed', argstr='--hp %f')
    ps = traits.Float(desc='To accelerate computations, preselection is used: if the normalized difference is above this threshold, the voxel will be discarded (non used for average)', argstr='--ps %f')
    inputVolume = File(position=-2, desc='Input MRI volume.', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output (filtered) MRI volume.', argstr='%s')