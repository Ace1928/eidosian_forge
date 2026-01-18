from ..base import File, TraitedSpec, traits, isdefined, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
class DwiToolInputSpec(CommandLineInputSpec):
    """Input Spec for DwiTool."""
    desc = 'The source image containing the fitted model.'
    source_file = File(position=1, exists=True, desc=desc, argstr='-source %s', mandatory=True)
    desc = 'The file containing the bvalues of the source DWI.'
    bval_file = File(position=2, exists=True, desc=desc, argstr='-bval %s', mandatory=True)
    desc = 'The file containing the bvectors of the source DWI.'
    bvec_file = File(position=3, exists=True, desc=desc, argstr='-bvec %s')
    b0_file = File(position=4, exists=True, desc='The B0 image corresponding to the source DWI', argstr='-b0 %s')
    mask_file = File(position=5, exists=True, desc='The image mask', argstr='-mask %s')
    desc = 'Filename of multi-compartment model parameter map (-ivim,-ball,-nod)'
    mcmap_file = File(name_source=['source_file'], name_template='%s_mcmap.nii.gz', desc=desc, argstr='-mcmap %s')
    desc = 'Filename of synthetic image. Requires: bvec_file/b0_file.'
    syn_file = File(name_source=['source_file'], name_template='%s_syn.nii.gz', desc=desc, argstr='-syn %s', requires=['bvec_file', 'b0_file'])
    mdmap_file = File(name_source=['source_file'], name_template='%s_mdmap.nii.gz', desc='Filename of MD map/ADC', argstr='-mdmap %s')
    famap_file = File(name_source=['source_file'], name_template='%s_famap.nii.gz', desc='Filename of FA map', argstr='-famap %s')
    v1map_file = File(name_source=['source_file'], name_template='%s_v1map.nii.gz', desc='Filename of PDD map [x,y,z]', argstr='-v1map %s')
    rgbmap_file = File(name_source=['source_file'], name_template='%s_rgbmap.nii.gz', desc='Filename of colour FA map.', argstr='-rgbmap %s')
    logdti_file = File(name_source=['source_file'], name_template='%s_logdti2.nii.gz', desc='Filename of output logdti map.', argstr='-logdti2 %s')
    desc = 'Input is a single exponential to non-directional data [default with no b-vectors]'
    mono_flag = traits.Bool(desc=desc, position=6, argstr='-mono', xor=['ivim_flag', 'dti_flag', 'dti_flag2', 'ball_flag', 'ballv_flag', 'nod_flag', 'nodv_flag'])
    desc = 'Inputs is an IVIM model to non-directional data.'
    ivim_flag = traits.Bool(desc=desc, position=6, argstr='-ivim', xor=['mono_flag', 'dti_flag', 'dti_flag2', 'ball_flag', 'ballv_flag', 'nod_flag', 'nodv_flag'])
    dti_flag = traits.Bool(desc='Input is a tensor model diag/off-diag.', position=6, argstr='-dti', xor=['mono_flag', 'ivim_flag', 'dti_flag2', 'ball_flag', 'ballv_flag', 'nod_flag', 'nodv_flag'])
    dti_flag2 = traits.Bool(desc='Input is a tensor model lower triangular', position=6, argstr='-dti2', xor=['mono_flag', 'ivim_flag', 'dti_flag', 'ball_flag', 'ballv_flag', 'nod_flag', 'nodv_flag'])
    ball_flag = traits.Bool(desc='Input is a ball and stick model.', position=6, argstr='-ball', xor=['mono_flag', 'ivim_flag', 'dti_flag', 'dti_flag2', 'ballv_flag', 'nod_flag', 'nodv_flag'])
    desc = 'Input is a ball and stick model with optimised PDD.'
    ballv_flag = traits.Bool(desc=desc, position=6, argstr='-ballv', xor=['mono_flag', 'ivim_flag', 'dti_flag', 'dti_flag2', 'ball_flag', 'nod_flag', 'nodv_flag'])
    nod_flag = traits.Bool(desc='Input is a NODDI model', position=6, argstr='-nod', xor=['mono_flag', 'ivim_flag', 'dti_flag', 'dti_flag2', 'ball_flag', 'ballv_flag', 'nodv_flag'])
    nodv_flag = traits.Bool(desc='Input is a NODDI model with optimised PDD', position=6, argstr='-nodv', xor=['mono_flag', 'ivim_flag', 'dti_flag', 'dti_flag2', 'ball_flag', 'ballv_flag', 'nod_flag'])
    diso_val = traits.Float(desc='Isotropic diffusivity for -nod [3e-3]', argstr='-diso %f')
    dpr_val = traits.Float(desc='Parallel diffusivity for -nod [1.7e-3].', argstr='-dpr %f')