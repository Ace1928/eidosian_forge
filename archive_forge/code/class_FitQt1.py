from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftyFitCommand
from ..niftyreg.base import get_custom_path
class FitQt1(NiftyFitCommand):
    """Interface for executable fit_qt1 from Niftyfit platform.

    Use NiftyFit to perform Qt1 fitting.

    T1 Fitting Routine (To inversion recovery or spgr data).
    Fits single component T1 maps in the first instance.

    `Source code <https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyFit-Release>`_

    Examples
    --------

    >>> from nipype.interfaces.niftyfit import FitQt1
    >>> fit_qt1 = FitQt1()
    >>> fit_qt1.inputs.source_file = 'TI4D.nii.gz'
    >>> fit_qt1.cmdline
    'fit_qt1 -source TI4D.nii.gz -comp TI4D_comp.nii.gz -error TI4D_error.nii.gz -m0map TI4D_m0map.nii.gz -mcmap TI4D_mcmap.nii.gz -res TI4D_res.nii.gz -syn TI4D_syn.nii.gz -t1map TI4D_t1map.nii.gz'

    """
    _cmd = get_custom_path('fit_qt1', env_dir='NIFTYFITDIR')
    input_spec = FitQt1InputSpec
    output_spec = FitQt1OutputSpec
    _suffix = '_fit_qt1'