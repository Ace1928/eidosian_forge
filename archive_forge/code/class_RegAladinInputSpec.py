import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegAladinInputSpec(NiftyRegCommandInputSpec):
    """Input Spec for RegAladin."""
    ref_file = File(exists=True, desc='The input reference/target image', argstr='-ref %s', mandatory=True)
    flo_file = File(exists=True, desc='The input floating/source image', argstr='-flo %s', mandatory=True)
    nosym_flag = traits.Bool(argstr='-noSym', desc='Turn off symmetric registration')
    rig_only_flag = traits.Bool(argstr='-rigOnly', desc='Do only a rigid registration')
    desc = 'Directly optimise the affine parameters'
    aff_direct_flag = traits.Bool(argstr='-affDirect', desc=desc)
    in_aff_file = File(exists=True, desc='The input affine transformation', argstr='-inaff %s')
    rmask_file = File(exists=True, desc='The input reference mask', argstr='-rmask %s')
    fmask_file = File(exists=True, desc='The input floating mask', argstr='-fmask %s')
    maxit_val = traits.Range(desc='Maximum number of iterations', argstr='-maxit %d', low=0)
    ln_val = traits.Range(desc='Number of resolution levels to create', argstr='-ln %d', low=0)
    lp_val = traits.Range(desc='Number of resolution levels to perform', argstr='-lp %d', low=0)
    desc = 'Amount of smoothing to apply to reference image'
    smoo_r_val = traits.Float(desc=desc, argstr='-smooR %f')
    desc = 'Amount of smoothing to apply to floating image'
    smoo_f_val = traits.Float(desc=desc, argstr='-smooF %f')
    desc = 'Use nifti header to initialise transformation'
    nac_flag = traits.Bool(desc=desc, argstr='-nac')
    desc = 'Use the masks centre of mass to initialise the transformation'
    cog_flag = traits.Bool(desc=desc, argstr='-cog')
    v_val = traits.Range(desc='Percent of blocks that are active', argstr='-pv %d', low=0)
    i_val = traits.Range(desc='Percent of inlier blocks', argstr='-pi %d', low=0)
    ref_low_val = traits.Float(desc='Lower threshold value on reference image', argstr='-refLowThr %f')
    ref_up_val = traits.Float(desc='Upper threshold value on reference image', argstr='-refUpThr %f')
    flo_low_val = traits.Float(desc='Lower threshold value on floating image', argstr='-floLowThr %f')
    flo_up_val = traits.Float(desc='Upper threshold value on floating image', argstr='-floUpThr %f')
    platform_val = traits.Int(desc='Platform index', argstr='-platf %i')
    gpuid_val = traits.Int(desc='Device to use id', argstr='-gpuid %i')
    verbosity_off_flag = traits.Bool(argstr='-voff', desc='Turn off verbose output')
    aff_file = File(name_source=['flo_file'], name_template='%s_aff.txt', desc='The output affine matrix file', argstr='-aff %s')
    res_file = File(name_source=['flo_file'], name_template='%s_res.nii.gz', desc='The affine transformed floating image', argstr='-res %s')