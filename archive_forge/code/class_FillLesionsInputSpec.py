import warnings
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class FillLesionsInputSpec(CommandLineInputSpec):
    """Input Spec for FillLesions."""
    in_file = File(argstr='-i %s', exists=True, mandatory=True, desc='Input image to fill lesions', position=1)
    lesion_mask = File(argstr='-l %s', exists=True, mandatory=True, desc='Lesion mask', position=2)
    out_file = File(name_source=['in_file'], name_template='%s_lesions_filled.nii.gz', desc='The output filename of the fill lesions results', argstr='-o %s', position=3)
    desc = 'Dilate the mask <int> times (in voxels, by default 0)'
    in_dilation = traits.Int(desc=desc, argstr='-dil %d')
    desc = 'Percentage of minimum number of voxels between patches <float> (by default 0.5).'
    match = traits.Float(desc=desc, argstr='-match %f')
    desc = 'Minimum percentage of valid voxels in target patch <float> (by default 0).'
    search = traits.Float(desc=desc, argstr='-search %f')
    desc = 'Smoothing by <float> (in minimal 6-neighbourhood voxels (by default 0.1)).'
    smooth = traits.Float(desc=desc, argstr='-smo %f')
    desc = 'Search regions size respect biggest patch size (by default 4).'
    size = traits.Int(desc=desc, argstr='-size %d')
    desc = 'Patch cardinality weighting factor (by default 2).'
    cwf = traits.Float(desc=desc, argstr='-cwf %f')
    desc = 'Give a binary mask with the valid search areas.'
    bin_mask = File(desc=desc, argstr='-mask %s')
    desc = "Guizard et al. (FIN 2015) method, it doesn't include the multiresolution/hierarchical inpainting part, this part needs to be done with some external software such as reg_tools and reg_resample from NiftyReg. By default it uses the method presented in Prados et al. (Neuroimage 2016)."
    other = traits.Bool(desc=desc, argstr='-other')
    use_2d = traits.Bool(desc='Uses 2D patches in the Z axis, by default 3D.', argstr='-2D')
    debug = traits.Bool(desc='Save all intermidium files (by default OFF).', argstr='-debug')
    desc = 'Set output <datatype> (char, short, int, uchar, ushort, uint, float, double).'
    out_datatype = traits.String(desc=desc, argstr='-odt %s')
    verbose = traits.Bool(desc='Verbose (by default OFF).', argstr='-v')