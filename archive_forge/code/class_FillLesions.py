import warnings
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class FillLesions(NiftySegCommand):
    """Interface for executable seg_FillLesions from NiftySeg platform.

    Fill all the masked lesions with WM intensity average.

    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`_ |
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyseg
    >>> node = niftyseg.FillLesions()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.lesion_mask = 'im2.nii'
    >>> node.cmdline
    'seg_FillLesions -i im1.nii -l im2.nii -o im1_lesions_filled.nii.gz'

    """
    _cmd = get_custom_path('seg_FillLesions', env_dir='NIFTYSEGDIR')
    input_spec = FillLesionsInputSpec
    output_spec = FillLesionsOutputSpec