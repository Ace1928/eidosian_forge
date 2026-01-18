import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FASTInputSpec(FSLCommandInputSpec):
    """Defines inputs (trait classes) for FAST"""
    in_files = InputMultiPath(File(exists=True), copyfile=False, desc='image, or multi-channel set of images, to be segmented', argstr='%s', position=-1, mandatory=True)
    out_basename = File(desc='base name of output files', argstr='-o %s')
    number_classes = traits.Range(low=1, high=10, argstr='-n %d', desc='number of tissue-type classes')
    output_biasfield = traits.Bool(desc='output estimated bias field', argstr='-b')
    output_biascorrected = traits.Bool(desc='output restored image (bias-corrected image)', argstr='-B')
    img_type = traits.Enum((1, 2, 3), desc='int specifying type of image: (1 = T1, 2 = T2, 3 = PD)', argstr='-t %d')
    bias_iters = traits.Range(low=1, high=10, argstr='-I %d', desc='number of main-loop iterations during bias-field removal')
    bias_lowpass = traits.Range(low=4, high=40, desc='bias field smoothing extent (FWHM) in mm', argstr='-l %d', units='mm')
    init_seg_smooth = traits.Range(low=0.0001, high=0.1, desc='initial segmentation spatial smoothness (during bias field estimation)', argstr='-f %.3f')
    segments = traits.Bool(desc='outputs a separate binary image for each tissue type', argstr='-g')
    init_transform = File(exists=True, desc='<standard2input.mat> initialise using priors', argstr='-a %s')
    other_priors = InputMultiPath(File(exist=True), desc='alternative prior images', argstr='-A %s', minlen=3, maxlen=3)
    no_pve = traits.Bool(desc='turn off PVE (partial volume estimation)', argstr='--nopve')
    no_bias = traits.Bool(desc='do not remove bias field', argstr='-N')
    use_priors = traits.Bool(desc='use priors throughout', argstr='-P')
    segment_iters = traits.Range(low=1, high=50, desc='number of segmentation-initialisation iterations', argstr='-W %d')
    mixel_smooth = traits.Range(low=0.0, high=1.0, desc='spatial smoothness for mixeltype', argstr='-R %.2f')
    iters_afterbias = traits.Range(low=1, high=20, desc='number of main-loop iterations after bias-field removal', argstr='-O %d')
    hyper = traits.Range(low=0.0, high=1.0, desc='segmentation spatial smoothness', argstr='-H %.2f')
    verbose = traits.Bool(desc='switch on diagnostic messages', argstr='-v')
    manual_seg = File(exists=True, desc='Filename containing intensities', argstr='-s %s')
    probability_maps = traits.Bool(desc='outputs individual probability maps', argstr='-p')