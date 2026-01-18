import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class BandpassInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dBandpass', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_bp', desc='output file from 3dBandpass', argstr='-prefix %s', position=1, name_source='in_file')
    lowpass = traits.Float(desc='lowpass', argstr='%f', position=-2, mandatory=True)
    highpass = traits.Float(desc='highpass', argstr='%f', position=-3, mandatory=True)
    mask = File(desc='mask file', position=2, argstr='-mask %s', exists=True)
    despike = traits.Bool(argstr='-despike', desc="Despike each time series before other processing. Hopefully, you don't actually need to do this, which is why it is optional.")
    orthogonalize_file = InputMultiPath(File(exists=True), argstr='-ort %s', desc="Also orthogonalize input to columns in f.1D. Multiple '-ort' options are allowed.")
    orthogonalize_dset = File(exists=True, argstr='-dsort %s', desc="Orthogonalize each voxel to the corresponding voxel time series in dataset 'fset', which must have the same spatial and temporal grid structure as the main input dataset. At present, only one '-dsort' option is allowed.")
    no_detrend = traits.Bool(argstr='-nodetrend', desc='Skip the quadratic detrending of the input that occurs before the FFT-based bandpassing. You would only want to do this if the dataset had been detrended already in some other program.')
    tr = traits.Float(argstr='-dt %f', desc='Set time step (TR) in sec [default=from dataset header].')
    nfft = traits.Int(argstr='-nfft %d', desc='Set the FFT length [must be a legal value].')
    normalize = traits.Bool(argstr='-norm', desc='Make all output time series have L2 norm = 1 (i.e., sum of squares = 1).')
    automask = traits.Bool(argstr='-automask', desc='Create a mask from the input dataset.')
    blur = traits.Float(argstr='-blur %f', desc="Blur (inside the mask only) with a filter width (FWHM) of 'fff' millimeters.")
    localPV = traits.Float(argstr='-localPV %f', desc="Replace each vector by the local Principal Vector (AKA first singular vector) from a neighborhood of radius 'rrr' millimeters. Note that the PV time series is L2 normalized. This option is mostly for Bob Cox to have fun with.")
    notrans = traits.Bool(argstr='-notrans', desc="Don't check for initial positive transients in the data. The test is a little slow, so skipping it is OK, if you KNOW the data time series are transient-free.")