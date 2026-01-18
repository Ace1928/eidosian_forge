from ..base import File, TraitedSpec, traits, isdefined, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
class FitDwiInputSpec(CommandLineInputSpec):
    """Input Spec for FitDwi."""
    source_file = File(position=1, exists=True, argstr='-source %s', mandatory=True, desc='The source image containing the dwi data.')
    desc = 'The file containing the bvalues of the source DWI.'
    bval_file = File(position=2, exists=True, argstr='-bval %s', mandatory=True, desc=desc)
    desc = 'The file containing the bvectors of the source DWI.'
    bvec_file = File(position=3, exists=True, argstr='-bvec %s', mandatory=True, desc=desc)
    te_file = File(exists=True, argstr='-TE %s', desc='Filename of TEs (ms).', xor=['te_file'])
    te_value = File(exists=True, argstr='-TE %s', desc='Value of TEs (ms).', xor=['te_file'])
    mask_file = File(exists=True, desc='The image mask', argstr='-mask %s')
    desc = 'Filename of parameter priors for -ball and -nod.'
    prior_file = File(exists=True, argstr='-prior %s', desc=desc)
    desc = 'Rotate the output tensors according to the q/s form of the image (resulting tensors will be in mm coordinates, default: 0).'
    rot_sform_flag = traits.Int(desc=desc, argstr='-rotsform %d')
    error_file = File(name_source=['source_file'], name_template='%s_error.nii.gz', desc='Filename of parameter error maps.', argstr='-error %s')
    res_file = File(name_source=['source_file'], name_template='%s_resmap.nii.gz', desc='Filename of model residual map.', argstr='-res %s')
    syn_file = File(name_source=['source_file'], name_template='%s_syn.nii.gz', desc='Filename of synthetic image.', argstr='-syn %s')
    nodiff_file = File(name_source=['source_file'], name_template='%s_no_diff.nii.gz', desc='Filename of average no diffusion image.', argstr='-nodiff %s')
    mcmap_file = File(name_source=['source_file'], name_template='%s_mcmap.nii.gz', desc='Filename of multi-compartment model parameter map (-ivim,-ball,-nod)', argstr='-mcmap %s', requires=['nodv_flag'])
    mdmap_file = File(name_source=['source_file'], name_template='%s_mdmap.nii.gz', desc='Filename of MD map/ADC', argstr='-mdmap %s')
    famap_file = File(name_source=['source_file'], name_template='%s_famap.nii.gz', desc='Filename of FA map', argstr='-famap %s')
    v1map_file = File(name_source=['source_file'], name_template='%s_v1map.nii.gz', desc='Filename of PDD map [x,y,z]', argstr='-v1map %s')
    rgbmap_file = File(name_source=['source_file'], name_template='%s_rgbmap.nii.gz', desc='Filename of colour-coded FA map', argstr='-rgbmap %s', requires=['dti_flag'])
    desc = 'Use lower triangular (tenmap2) or diagonal, off-diagonal tensor format'
    ten_type = traits.Enum('lower-tri', 'diag-off-diag', desc=desc, usedefault=True)
    tenmap_file = File(name_source=['source_file'], name_template='%s_tenmap.nii.gz', desc='Filename of tensor map [diag,offdiag].', argstr='-tenmap %s', requires=['dti_flag'])
    tenmap2_file = File(name_source=['source_file'], name_template='%s_tenmap2.nii.gz', desc='Filename of tensor map [lower tri]', argstr='-tenmap2 %s', requires=['dti_flag'])
    desc = 'Fit single exponential to non-directional data [default with no b-vectors]'
    mono_flag = traits.Bool(desc=desc, argstr='-mono', position=4, xor=['ivim_flag', 'dti_flag', 'ball_flag', 'ballv_flag', 'nod_flag', 'nodv_flag'])
    ivim_flag = traits.Bool(desc='Fit IVIM model to non-directional data.', argstr='-ivim', position=4, xor=['mono_flag', 'dti_flag', 'ball_flag', 'ballv_flag', 'nod_flag', 'nodv_flag'])
    desc = 'Fit the tensor model [default with b-vectors].'
    dti_flag = traits.Bool(desc=desc, argstr='-dti', position=4, xor=['mono_flag', 'ivim_flag', 'ball_flag', 'ballv_flag', 'nod_flag', 'nodv_flag'])
    ball_flag = traits.Bool(desc='Fit the ball and stick model.', argstr='-ball', position=4, xor=['mono_flag', 'ivim_flag', 'dti_flag', 'ballv_flag', 'nod_flag', 'nodv_flag'])
    desc = 'Fit the ball and stick model with optimised PDD.'
    ballv_flag = traits.Bool(desc=desc, argstr='-ballv', position=4, xor=['mono_flag', 'ivim_flag', 'dti_flag', 'ball_flag', 'nod_flag', 'nodv_flag'])
    nod_flag = traits.Bool(desc='Fit the NODDI model', argstr='-nod', position=4, xor=['mono_flag', 'ivim_flag', 'dti_flag', 'ball_flag', 'ballv_flag', 'nodv_flag'])
    nodv_flag = traits.Bool(desc='Fit the NODDI model with optimised PDD', argstr='-nodv', position=4, xor=['mono_flag', 'ivim_flag', 'dti_flag', 'ball_flag', 'ballv_flag', 'nod_flag'])
    desc = 'Maximum number of non-linear LSQR iterations [100x2 passes])'
    maxit_val = traits.Int(desc=desc, argstr='-maxit %d', requires=['gn_flag'])
    desc = 'LM parameters (initial value, decrease rate) [100,1.2].'
    lm_vals = traits.Tuple(traits.Float, traits.Float, argstr='-lm %f %f', requires=['gn_flag'], desc=desc)
    desc = 'Use Gauss-Newton algorithm [Levenberg-Marquardt].'
    gn_flag = traits.Bool(desc=desc, argstr='-gn', xor=['wls_flag'])
    desc = 'Use Variational Bayes fitting with known prior (currently identity covariance...).'
    vb_flag = traits.Bool(desc=desc, argstr='-vb')
    cov_file = File(exists=True, desc='Filename of ithe nc*nc covariance matrix [I]', argstr='-cov %s')
    wls_flag = traits.Bool(desc=desc, argstr='-wls', xor=['gn_flag'])
    desc = 'Use location-weighted least squares for DTI fitting [3x3 Gaussian]'
    swls_val = traits.Float(desc=desc, argstr='-swls %f')
    slice_no = traits.Int(desc='Fit to single slice number.', argstr='-slice %d')
    voxel = traits.Tuple(traits.Int, traits.Int, traits.Int, desc='Fit to single voxel only.', argstr='-voxel %d %d %d')
    diso_val = traits.Float(desc='Isotropic diffusivity for -nod [3e-3]', argstr='-diso %f')
    dpr_val = traits.Float(desc='Parallel diffusivity for -nod [1.7e-3].', argstr='-dpr %f')
    wm_t2_val = traits.Float(desc='White matter T2 value [80ms].', argstr='-wmT2 %f')
    csf_t2_val = traits.Float(desc='CSF T2 value [400ms].', argstr='-csfT2 %f')
    desc = 'Threshold for perfusion/diffsuion effects [100].'
    perf_thr = traits.Float(desc=desc, argstr='-perfthreshold %f')
    mcout = File(name_source=['source_file'], name_template='%s_mcout.txt', desc='Filename of mc samples (ascii text file)', argstr='-mcout %s')
    mcsamples = traits.Int(desc='Number of samples to keep [100].', argstr='-mcsamples %d')
    mcmaxit = traits.Int(desc='Number of iterations to run [10,000].', argstr='-mcmaxit %d')
    acceptance = traits.Float(desc='Fraction of iterations to accept [0.23].', argstr='-accpetance %f')