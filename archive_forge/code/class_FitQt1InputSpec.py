from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
class FitQt1InputSpec(CommandLineInputSpec):
    """Input Spec for FitQt1."""
    desc = 'Filename of the 4D Multi-Echo T1 source image.'
    source_file = File(position=1, exists=True, desc=desc, argstr='-source %s', mandatory=True)
    t1map_file = File(name_source=['source_file'], name_template='%s_t1map.nii.gz', argstr='-t1map %s', desc='Filename of the estimated output T1 map (in ms).')
    m0map_file = File(name_source=['source_file'], name_template='%s_m0map.nii.gz', argstr='-m0map %s', desc='Filename of the estimated input M0 map.')
    desc = 'Filename of the estimated output multi-parameter map.'
    mcmap_file = File(name_source=['source_file'], name_template='%s_mcmap.nii.gz', argstr='-mcmap %s', desc=desc)
    comp_file = File(name_source=['source_file'], name_template='%s_comp.nii.gz', argstr='-comp %s', desc='Filename of the estimated multi-component T1 map.')
    desc = 'Filename of the error map (symmetric matrix, [Diag,OffDiag]).'
    error_file = File(name_source=['source_file'], name_template='%s_error.nii.gz', argstr='-error %s', desc=desc)
    syn_file = File(name_source=['source_file'], name_template='%s_syn.nii.gz', argstr='-syn %s', desc='Filename of the synthetic ASL data.')
    res_file = File(name_source=['source_file'], name_template='%s_res.nii.gz', argstr='-res %s', desc='Filename of the model fit residuals')
    mask = File(position=2, exists=True, desc='Filename of image mask.', argstr='-mask %s')
    prior = File(position=3, exists=True, desc='Filename of parameter prior.', argstr='-prior %s')
    te_value = traits.Float(desc='TE Echo Time [0ms!].', argstr='-TE %f', position=4)
    tr_value = traits.Float(desc='TR Repetition Time [10s!].', argstr='-TR %f', position=5)
    desc = 'Number of components to fit [1] (currently IR/SR only)'
    nb_comp = traits.Int(desc=desc, position=6, argstr='-nc %d')
    desc = 'Set LM parameters (initial value, decrease rate) [100,1.2].'
    lm_val = traits.Tuple(traits.Float, traits.Float, desc=desc, argstr='-lm %f %f', position=7)
    desc = 'Use Gauss-Newton algorithm [Levenberg-Marquardt].'
    gn_flag = traits.Bool(desc=desc, argstr='-gn', position=8)
    slice_no = traits.Int(desc='Fit to single slice number.', argstr='-slice %d', position=9)
    voxel = traits.Tuple(traits.Int, traits.Int, traits.Int, desc='Fit to single voxel only.', argstr='-voxel %d %d %d', position=10)
    maxit = traits.Int(desc='NLSQR iterations [100].', argstr='-maxit %d', position=11)
    sr_flag = traits.Bool(desc='Saturation Recovery fitting [default].', argstr='-SR', position=12)
    ir_flag = traits.Bool(desc='Inversion Recovery fitting [default].', argstr='-IR', position=13)
    tis = traits.List(traits.Float, position=14, desc='Inversion times for T1 data [1s,2s,5s].', argstr='-TIs %s', sep=' ')
    tis_list = File(exists=True, argstr='-TIlist %s', desc='Filename of list of pre-defined TIs.')
    t1_list = File(exists=True, argstr='-T1list %s', desc='Filename of list of pre-defined T1s')
    t1min = traits.Float(desc='Minimum tissue T1 value [400ms].', argstr='-T1min %f')
    t1max = traits.Float(desc='Maximum tissue T1 value [4000ms].', argstr='-T1max %f')
    spgr = traits.Bool(desc='Spoiled Gradient Echo fitting', argstr='-SPGR')
    flips = traits.List(traits.Float, desc='Flip angles', argstr='-flips %s', sep=' ')
    desc = 'Filename of list of pre-defined flip angles (deg).'
    flips_list = File(exists=True, argstr='-fliplist %s', desc=desc)
    desc = 'Filename of B1 estimate for fitting (or include in prior).'
    b1map = File(exists=True, argstr='-b1map %s', desc=desc)
    mcout = File(exists=True, desc='Filename of mc samples (ascii text file)', argstr='-mcout %s')
    mcsamples = traits.Int(desc='Number of samples to keep [100].', argstr='-mcsamples %d')
    mcmaxit = traits.Int(desc='Number of iterations to run [10,000].', argstr='-mcmaxit %d')
    acceptance = traits.Float(desc='Fraction of iterations to accept [0.23].', argstr='-acceptance %f')