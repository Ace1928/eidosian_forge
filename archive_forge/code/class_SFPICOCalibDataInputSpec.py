import os
from ...utils.filemanip import split_filename
from ..base import (
class SFPICOCalibDataInputSpec(StdOutCommandLineInputSpec):
    snr = traits.Float(argstr='-snr %f', units='NA', desc='Specifies  the  signal-to-noise ratio of the non-diffusion-weighted measurements to use in simulations.')
    scheme_file = File(exists=True, argstr='-schemefile %s', mandatory=True, desc='Specifies the scheme file for the diffusion MRI data')
    info_file = File(desc='The name to be given to the information output filename.', argstr='-infooutputfile %s', mandatory=True, genfile=True, hash_files=False)
    trace = traits.Float(argstr='-trace %f', units='NA', desc='Trace of the diffusion tensor(s) used in the test function.')
    onedtfarange = traits.List(traits.Float, argstr='-onedtfarange %s', minlen=2, maxlen=2, units='NA', desc='Minimum and maximum FA for the single tensor synthetic data.')
    onedtfastep = traits.Float(argstr='-onedtfastep %f', units='NA', desc='FA step size controlling how many steps there are between the minimum and maximum FA settings.')
    twodtfarange = traits.List(traits.Float, argstr='-twodtfarange %s', minlen=2, maxlen=2, units='NA', desc='Minimum and maximum FA for the two tensor synthetic data. FA is varied for both tensors to give all the different permutations.')
    twodtfastep = traits.Float(argstr='-twodtfastep %f', units='NA', desc='FA step size controlling how many steps there are between the minimum and maximum FA settings for the two tensor cases.')
    twodtanglerange = traits.List(traits.Float, argstr='-twodtanglerange %s', minlen=2, maxlen=2, units='NA', desc='Minimum and maximum crossing angles between the two fibres.')
    twodtanglestep = traits.Float(argstr='-twodtanglestep %f', units='NA', desc='Angle step size controlling how many steps there are between the minimum and maximum crossing angles for the two tensor cases.')
    twodtmixmax = traits.Float(argstr='-twodtmixmax %f', units='NA', desc='Mixing parameter controlling the proportion of one fibre population to the other. The minimum mixing parameter is (1 - twodtmixmax).')
    twodtmixstep = traits.Float(argstr='-twodtmixstep %f', units='NA', desc='Mixing parameter step size for the two tensor cases. Specify how many mixing parameter increments to use.')
    seed = traits.Float(argstr='-seed %f', units='NA', desc='Specifies the random seed to use for noise generation in simulation trials.')