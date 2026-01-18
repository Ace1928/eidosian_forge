import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FSLXCommandInputSpec(FSLCommandInputSpec):
    dwi = File(exists=True, argstr='--data=%s', mandatory=True, desc='diffusion weighted image data file')
    mask = File(exists=True, argstr='--mask=%s', mandatory=True, desc='brain binary mask file (i.e. from BET)')
    bvecs = File(exists=True, argstr='--bvecs=%s', mandatory=True, desc='b vectors file')
    bvals = File(exists=True, argstr='--bvals=%s', mandatory=True, desc='b values file')
    logdir = Directory('.', argstr='--logdir=%s', usedefault=True)
    n_fibres = traits.Range(usedefault=True, low=1, value=2, argstr='--nfibres=%d', desc='Maximum number of fibres to fit in each voxel', mandatory=True)
    model = traits.Enum(1, 2, 3, argstr='--model=%d', desc='use monoexponential (1, default, required for single-shell) or multiexponential (2, multi-shell) model')
    fudge = traits.Int(argstr='--fudge=%d', desc='ARD fudge factor')
    n_jumps = traits.Int(5000, usedefault=True, argstr='--njumps=%d', desc='Num of jumps to be made by MCMC')
    burn_in = traits.Range(low=0, value=0, usedefault=True, argstr='--burnin=%d', desc='Total num of jumps at start of MCMC to be discarded')
    burn_in_no_ard = traits.Range(low=0, value=0, usedefault=True, argstr='--burnin_noard=%d', desc='num of burnin jumps before the ard is imposed')
    sample_every = traits.Range(low=0, value=1, usedefault=True, argstr='--sampleevery=%d', desc='Num of jumps for each sample (MCMC)')
    update_proposal_every = traits.Range(low=1, value=40, usedefault=True, argstr='--updateproposalevery=%d', desc='Num of jumps for each update to the proposal density std (MCMC)')
    seed = traits.Int(argstr='--seed=%d', desc='seed for pseudo random number generator')
    _xor_inputs1 = ('no_ard', 'all_ard')
    no_ard = traits.Bool(argstr='--noard', xor=_xor_inputs1, desc='Turn ARD off on all fibres')
    all_ard = traits.Bool(argstr='--allard', xor=_xor_inputs1, desc='Turn ARD on on all fibres')
    _xor_inputs2 = ('no_spat', 'non_linear', 'cnlinear')
    no_spat = traits.Bool(argstr='--nospat', xor=_xor_inputs2, desc='Initialise with tensor, not spatially')
    non_linear = traits.Bool(argstr='--nonlinear', xor=_xor_inputs2, desc='Initialise with nonlinear fitting')
    cnlinear = traits.Bool(argstr='--cnonlinear', xor=_xor_inputs2, desc='Initialise with constrained nonlinear fitting')
    rician = traits.Bool(argstr='--rician', desc='use Rician noise modeling')
    _xor_inputs3 = ['f0_noard', 'f0_ard']
    f0_noard = traits.Bool(argstr='--f0', xor=_xor_inputs3, desc='Noise floor model: add to the model an unattenuated signal compartment f0')
    f0_ard = traits.Bool(argstr='--f0 --ardf0', xor=_xor_inputs3 + ['all_ard'], desc='Noise floor model: add to the model an unattenuated signal compartment f0')
    force_dir = traits.Bool(True, argstr='--forcedir', usedefault=True, desc='use the actual directory name given (do not add + to make a new directory)')