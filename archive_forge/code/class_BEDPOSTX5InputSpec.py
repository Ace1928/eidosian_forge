import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class BEDPOSTX5InputSpec(FSLXCommandInputSpec):
    dwi = File(exists=True, desc='diffusion weighted image data file', mandatory=True)
    mask = File(exists=True, desc='bet binary mask file', mandatory=True)
    bvecs = File(exists=True, desc='b vectors file', mandatory=True)
    bvals = File(exists=True, desc='b values file', mandatory=True)
    logdir = Directory(argstr='--logdir=%s')
    n_fibres = traits.Range(usedefault=True, low=1, value=2, argstr='-n %d', desc='Maximum number of fibres to fit in each voxel', mandatory=True)
    model = traits.Enum(1, 2, 3, argstr='-model %d', desc='use monoexponential (1, default, required for single-shell) or multiexponential (2, multi-shell) model')
    fudge = traits.Int(argstr='-w %d', desc='ARD fudge factor')
    n_jumps = traits.Int(5000, usedefault=True, argstr='-j %d', desc='Num of jumps to be made by MCMC')
    burn_in = traits.Range(low=0, value=0, usedefault=True, argstr='-b %d', desc='Total num of jumps at start of MCMC to be discarded')
    sample_every = traits.Range(low=0, value=1, usedefault=True, argstr='-s %d', desc='Num of jumps for each sample (MCMC)')
    out_dir = Directory('bedpostx', mandatory=True, desc='output directory', usedefault=True, position=1, argstr='%s')
    gradnonlin = traits.Bool(False, argstr='-g', desc='consider gradient nonlinearities, default off')
    grad_dev = File(exists=True, desc='grad_dev file, if gradnonlin, -g is True')
    use_gpu = traits.Bool(False, desc='Use the GPU version of bedpostx')