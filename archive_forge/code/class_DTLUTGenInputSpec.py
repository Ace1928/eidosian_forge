import os
from ...utils.filemanip import split_filename
from ..base import (
class DTLUTGenInputSpec(StdOutCommandLineInputSpec):
    lrange = traits.List(traits.Float, desc='Index to one-tensor LUTs. This is the ratio L1/L3 and L2 / L3.The LUT is square, with half the values calculated (because L2 / L3 cannot be less than L1 / L3 by definition).The minimum must be >= 1. For comparison, a ratio L1 / L3 = 10 with L2 / L3 = 1 corresponds to an FA of 0.891, and L1 / L3 = 15 with L2 / L3 = 1 corresponds to an FA of 0.929. The default range is 1 to 10.', argstr='-lrange %s', minlen=2, maxlen=2, position=1, units='NA')
    frange = traits.List(traits.Float, desc='Index to two-tensor LUTs. This is the fractional anisotropy of the two tensors. The default is 0.3 to 0.94', argstr='-frange %s', minlen=2, maxlen=2, position=1, units='NA')
    step = traits.Float(argstr='-step %f', units='NA', desc='Distance between points in the LUT.For example, if lrange is 1 to 10 and the step is 0.1, LUT entries will be computed at L1 / L3 = 1, 1.1, 1.2 ... 10.0 and at L2 / L3 = 1.0, 1.1 ... L1 / L3.For single tensor LUTs, the default step is 0.2, for two-tensor LUTs it is 0.02.')
    samples = traits.Int(argstr='-samples %d', units='NA', desc='The number of synthetic measurements to generate at each point in the LUT. The default is 2000.')
    snr = traits.Float(argstr='-snr %f', units='NA', desc='The signal to noise ratio of the unweighted (q = 0) measurements.This should match the SNR (in white matter) of the images that the LUTs are used with.')
    bingham = traits.Bool(argstr='-bingham', desc='Compute a LUT for the Bingham PDF. This is the default.')
    acg = traits.Bool(argstr='-acg', desc='Compute a LUT for the ACG PDF.')
    watson = traits.Bool(argstr='-watson', desc='Compute a LUT for the Watson PDF.')
    inversion = traits.Int(argstr='-inversion %d', units='NA', desc='Index of the inversion to use. The default is 1 (linear single tensor inversion).')
    trace = traits.Float(argstr='-trace %G', units='NA', desc='Trace of the diffusion tensor(s) used in the test function in the LUT generation. The default is 2100E-12 m^2 s^-1.')
    scheme_file = File(argstr='-schemefile %s', mandatory=True, position=2, desc='The scheme file of the images to be processed using this LUT.')