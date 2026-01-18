import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MCFLIRTInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, position=0, argstr='-in %s', mandatory=True, desc='timeseries to motion-correct')
    out_file = File(argstr='-out %s', genfile=True, desc='file to write', hash_files=False)
    cost = traits.Enum('mutualinfo', 'woods', 'corratio', 'normcorr', 'normmi', 'leastsquares', argstr='-cost %s', desc='cost function to optimize')
    bins = traits.Int(argstr='-bins %d', desc='number of histogram bins')
    dof = traits.Int(argstr='-dof %d', desc='degrees of freedom for the transformation')
    ref_vol = traits.Int(argstr='-refvol %d', desc='volume to align frames to')
    scaling = traits.Float(argstr='-scaling %.2f', desc='scaling factor to use')
    smooth = traits.Float(argstr='-smooth %.2f', desc='smoothing factor for the cost function')
    rotation = traits.Int(argstr='-rotation %d', desc='scaling factor for rotation tolerances')
    stages = traits.Int(argstr='-stages %d', desc='stages (if 4, perform final search with sinc interpolation')
    init = File(exists=True, argstr='-init %s', desc='initial transformation matrix')
    interpolation = traits.Enum('spline', 'nn', 'sinc', argstr='-%s_final', desc='interpolation method for transformation')
    use_gradient = traits.Bool(argstr='-gdt', desc='run search on gradient images')
    use_contour = traits.Bool(argstr='-edge', desc='run search on contour images')
    mean_vol = traits.Bool(argstr='-meanvol', desc='register to mean volume')
    stats_imgs = traits.Bool(argstr='-stats', desc='produce variance and std. dev. images')
    save_mats = traits.Bool(argstr='-mats', desc='save transformation matrices')
    save_plots = traits.Bool(argstr='-plots', desc='save transformation parameters')
    save_rms = traits.Bool(argstr='-rmsabs -rmsrel', desc='save rms displacement parameters')
    ref_file = File(exists=True, argstr='-reffile %s', desc='target image for motion correction')