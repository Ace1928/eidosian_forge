import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class RobustRegisterInputSpec(FSTraitedSpec):
    source_file = File(exists=True, mandatory=True, argstr='--mov %s', desc='volume to be registered')
    target_file = File(exists=True, mandatory=True, argstr='--dst %s', desc='target volume for the registration')
    out_reg_file = traits.Either(True, File, default=True, usedefault=True, argstr='--lta %s', desc='registration file; either True or filename')
    registered_file = traits.Either(traits.Bool, File, argstr='--warp %s', desc='registered image; either True or filename')
    weights_file = traits.Either(traits.Bool, File, argstr='--weights %s', desc='weights image to write; either True or filename')
    est_int_scale = traits.Bool(argstr='--iscale', desc='estimate intensity scale (recommended for unnormalized images)')
    trans_only = traits.Bool(argstr='--transonly', desc='find 3 parameter translation only')
    in_xfm_file = File(exists=True, argstr='--transform', desc='use initial transform on source')
    half_source = traits.Either(traits.Bool, File, argstr='--halfmov %s', desc='write source volume mapped to halfway space')
    half_targ = traits.Either(traits.Bool, File, argstr='--halfdst %s', desc='write target volume mapped to halfway space')
    half_weights = traits.Either(traits.Bool, File, argstr='--halfweights %s', desc='write weights volume mapped to halfway space')
    half_source_xfm = traits.Either(traits.Bool, File, argstr='--halfmovlta %s', desc='write transform from source to halfway space')
    half_targ_xfm = traits.Either(traits.Bool, File, argstr='--halfdstlta %s', desc='write transform from target to halfway space')
    auto_sens = traits.Bool(argstr='--satit', xor=['outlier_sens'], mandatory=True, desc='auto-detect good sensitivity')
    outlier_sens = traits.Float(argstr='--sat %.4f', xor=['auto_sens'], mandatory=True, desc='set outlier sensitivity explicitly')
    least_squares = traits.Bool(argstr='--leastsquares', desc='use least squares instead of robust estimator')
    no_init = traits.Bool(argstr='--noinit', desc='skip transform init')
    init_orient = traits.Bool(argstr='--initorient', desc='use moments for initial orient (recommended for stripped brains)')
    max_iterations = traits.Int(argstr='--maxit %d', desc='maximum # of times on each resolution')
    high_iterations = traits.Int(argstr='--highit %d', desc='max # of times on highest resolution')
    iteration_thresh = traits.Float(argstr='--epsit %.3f', desc='stop iterations when below threshold')
    subsample_thresh = traits.Int(argstr='--subsample %d', desc='subsample if dimension is above threshold size')
    outlier_limit = traits.Float(argstr='--wlimit %.3f', desc='set maximal outlier limit in satit')
    write_vo2vox = traits.Bool(argstr='--vox2vox', desc='output vox2vox matrix (default is RAS2RAS)')
    no_multi = traits.Bool(argstr='--nomulti', desc='work on highest resolution')
    mask_source = File(exists=True, argstr='--maskmov %s', desc='image to mask source volume with')
    mask_target = File(exists=True, argstr='--maskdst %s', desc='image to mask target volume with')
    force_double = traits.Bool(argstr='--doubleprec', desc='use double-precision intensities')
    force_float = traits.Bool(argstr='--floattype', desc='use float intensities')