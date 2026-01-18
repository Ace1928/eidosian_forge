from ..base import TraitedSpec, File, traits, CommandLineInputSpec, InputMultiPath
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class EM(NiftySegCommand):
    """Interface for executable seg_EM from NiftySeg platform.

    seg_EM is a general purpose intensity based image segmentation tool. In
    it's simplest form, it takes in one 2D or 3D image and segments it in n
    classes.

    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`_ |
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyseg
    >>> node = niftyseg.EM()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.no_prior = 4
    >>> node.cmdline
    'seg_EM -in im1.nii -bc_order 3 -bc_thresh 0 -max_iter 100 -min_iter 0 -nopriors 4 -bc_out im1_bc_em.nii.gz -out im1_em.nii.gz -out_outlier im1_outlier_em.nii.gz'

    """
    _cmd = get_custom_path('seg_EM', env_dir='NIFTYSEGDIR')
    _suffix = '_em'
    input_spec = EMInputSpec
    output_spec = EMOutputSpec

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for seg_EM."""
        if opt == 'priors':
            _nb_priors = len(self.inputs.priors)
            return '-priors %d %s' % (_nb_priors, ' '.join(self.inputs.priors))
        else:
            return super(EM, self)._format_arg(opt, spec, val)