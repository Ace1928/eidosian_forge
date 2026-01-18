import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegF3DInputSpec(NiftyRegCommandInputSpec):
    """Input Spec for RegF3D."""
    ref_file = File(exists=True, desc='The input reference/target image', argstr='-ref %s', mandatory=True)
    flo_file = File(exists=True, desc='The input floating/source image', argstr='-flo %s', mandatory=True)
    aff_file = File(exists=True, desc='The input affine transformation file', argstr='-aff %s')
    incpp_file = File(exists=True, desc='The input cpp transformation file', argstr='-incpp %s')
    rmask_file = File(exists=True, desc='Reference image mask', argstr='-rmask %s')
    desc = 'Smoothing kernel width for reference image'
    ref_smooth_val = traits.Float(desc=desc, argstr='-smooR %f')
    desc = 'Smoothing kernel width for floating image'
    flo_smooth_val = traits.Float(desc=desc, argstr='-smooF %f')
    rlwth_thr_val = traits.Float(desc='Lower threshold for reference image', argstr='--rLwTh %f')
    rupth_thr_val = traits.Float(desc='Upper threshold for reference image', argstr='--rUpTh %f')
    flwth_thr_val = traits.Float(desc='Lower threshold for floating image', argstr='--fLwTh %f')
    fupth_thr_val = traits.Float(desc='Upper threshold for floating image', argstr='--fUpTh %f')
    desc = 'Lower threshold for reference image at the specified time point'
    rlwth2_thr_val = traits.Tuple(traits.Range(low=0), traits.Float, desc=desc, argstr='-rLwTh %d %f')
    desc = 'Upper threshold for reference image at the specified time point'
    rupth2_thr_val = traits.Tuple(traits.Range(low=0), traits.Float, desc=desc, argstr='-rUpTh %d %f')
    desc = 'Lower threshold for floating image at the specified time point'
    flwth2_thr_val = traits.Tuple(traits.Range(low=0), traits.Float, desc=desc, argstr='-fLwTh %d %f')
    desc = 'Upper threshold for floating image at the specified time point'
    fupth2_thr_val = traits.Tuple(traits.Range(low=0), traits.Float, desc=desc, argstr='-fUpTh %d %f')
    sx_val = traits.Float(desc='Final grid spacing along the x axes', argstr='-sx %f')
    sy_val = traits.Float(desc='Final grid spacing along the y axes', argstr='-sy %f')
    sz_val = traits.Float(desc='Final grid spacing along the z axes', argstr='-sz %f')
    be_val = traits.Float(desc='Bending energy value', argstr='-be %f')
    le_val = traits.Float(desc='Linear elasticity penalty term', argstr='-le %f')
    jl_val = traits.Float(desc='Log of jacobian of deformation penalty value', argstr='-jl %f')
    desc = 'Do not approximate the log of jacobian penalty at control points only'
    no_app_jl_flag = traits.Bool(argstr='-noAppJL', desc=desc)
    desc = 'use NMI even when other options are specified'
    nmi_flag = traits.Bool(argstr='--nmi', desc=desc)
    desc = 'Number of bins in the histogram for reference image'
    rbn_val = traits.Range(low=0, desc=desc, argstr='--rbn %d')
    desc = 'Number of bins in the histogram for reference image'
    fbn_val = traits.Range(low=0, desc=desc, argstr='--fbn %d')
    desc = 'Number of bins in the histogram for reference image for given time point'
    rbn2_val = traits.Tuple(traits.Range(low=0), traits.Range(low=0), desc=desc, argstr='-rbn %d %d')
    desc = 'Number of bins in the histogram for reference image for given time point'
    fbn2_val = traits.Tuple(traits.Range(low=0), traits.Range(low=0), desc=desc, argstr='-fbn %d %d')
    lncc_val = traits.Float(desc='SD of the Gaussian for computing LNCC', argstr='--lncc %f')
    desc = 'SD of the Gaussian for computing LNCC for a given time point'
    lncc2_val = traits.Tuple(traits.Range(low=0), traits.Float, desc=desc, argstr='-lncc %d %f')
    ssd_flag = traits.Bool(desc='Use SSD as the similarity measure', argstr='--ssd')
    desc = 'Use SSD as the similarity measure for a given time point'
    ssd2_flag = traits.Range(low=0, desc=desc, argstr='-ssd %d')
    kld_flag = traits.Bool(desc='Use KL divergence as the similarity measure', argstr='--kld')
    desc = 'Use KL divergence as the similarity measure for a given time point'
    kld2_flag = traits.Range(low=0, desc=desc, argstr='-kld %d')
    amc_flag = traits.Bool(desc='Use additive NMI', argstr='-amc')
    nox_flag = traits.Bool(desc="Don't optimise in x direction", argstr='-nox')
    noy_flag = traits.Bool(desc="Don't optimise in y direction", argstr='-noy')
    noz_flag = traits.Bool(desc="Don't optimise in z direction", argstr='-noz')
    maxit_val = traits.Range(low=0, argstr='-maxit %d', desc='Maximum number of iterations per level')
    ln_val = traits.Range(low=0, argstr='-ln %d', desc='Number of resolution levels to create')
    lp_val = traits.Range(low=0, argstr='-lp %d', desc='Number of resolution levels to perform')
    nopy_flag = traits.Bool(desc='Do not use the multiresolution approach', argstr='-nopy')
    noconj_flag = traits.Bool(desc='Use simple GD optimization', argstr='-noConj')
    desc = 'Add perturbation steps after each optimization step'
    pert_val = traits.Range(low=0, desc=desc, argstr='-pert %d')
    vel_flag = traits.Bool(desc='Use velocity field integration', argstr='-vel')
    fmask_file = File(exists=True, desc='Floating image mask', argstr='-fmask %s')
    desc = 'Kernel width for smoothing the metric gradient'
    smooth_grad_val = traits.Float(desc=desc, argstr='-smoothGrad %f')
    pad_val = traits.Float(desc='Padding value', argstr='-pad %f')
    verbosity_off_flag = traits.Bool(argstr='-voff', desc='Turn off verbose output')
    cpp_file = File(name_source=['flo_file'], name_template='%s_cpp.nii.gz', desc='The output CPP file', argstr='-cpp %s')
    res_file = File(name_source=['flo_file'], name_template='%s_res.nii.gz', desc='The output resampled image', argstr='-res %s')