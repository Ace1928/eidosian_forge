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
class MNIBiasCorrectionInputSpec(FSTraitedSpec):
    in_file = File(exists=True, mandatory=True, argstr='--i %s', desc='input volume. Input can be any format accepted by mri_convert.')
    out_file = File(argstr='--o %s', name_source=['in_file'], name_template='%s_output', hash_files=False, keep_extension=True, desc='output volume. Output can be any format accepted by mri_convert. ' + 'If the output format is COR, then the directory must exist.')
    iterations = traits.Int(4, usedefault=True, argstr='--n %d', desc='Number of iterations to run nu_correct. Default is 4. This is the number of times ' + 'that nu_correct is repeated (ie, using the output from the previous run as the input for ' + 'the next). This is different than the -iterations option to nu_correct.')
    protocol_iterations = traits.Int(argstr='--proto-iters %d', desc='Passes Np as argument of the -iterations flag of nu_correct. This is different ' + 'than the --n flag above. Default is not to pass nu_correct the -iterations flag.')
    distance = traits.Int(argstr='--distance %d', desc='N3 -distance option')
    no_rescale = traits.Bool(argstr='--no-rescale', desc='do not rescale so that global mean of output == input global mean')
    mask = File(exists=True, argstr='--mask %s', desc='brainmask volume. Input can be any format accepted by mri_convert.')
    transform = File(exists=True, argstr='--uchar %s', desc='tal.xfm. Use mri_make_uchar instead of conforming')
    stop = traits.Float(argstr='--stop %f', desc='Convergence threshold below which iteration stops (suggest 0.01 to 0.0001)')
    shrink = traits.Int(argstr='--shrink %d', desc='Shrink parameter for finer sampling (default is 4)')