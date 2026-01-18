import warnings
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class PatchMatchInputSpec(CommandLineInputSpec):
    """Input Spec for PatchMatch."""
    in_file = File(argstr='-i %s', exists=True, mandatory=True, desc='Input image to segment', position=1)
    mask_file = File(argstr='-m %s', exists=True, mandatory=True, desc='Input mask for the area where applies PatchMatch', position=2)
    database_file = File(argstr='-db %s', exists=True, mandatory=True, desc='Database with the segmentations', position=3)
    out_file = File(name_source=['in_file'], name_template='%s_pm.nii.gz', desc='The output filename of the patchmatch results', argstr='-o %s', position=4)
    patch_size = traits.Int(desc='Patch size, #voxels', argstr='-size %i')
    desc = 'Constrained search area size, number of times bigger than the patchsize'
    cs_size = traits.Int(desc=desc, argstr='-cs %i')
    match_num = traits.Int(desc='Number of better matching', argstr='-match %i')
    pm_num = traits.Int(desc='Number of patchmatch executions', argstr='-pm %i')
    desc = 'Number of iterations for the patchmatch algorithm'
    it_num = traits.Int(desc=desc, argstr='-it %i')