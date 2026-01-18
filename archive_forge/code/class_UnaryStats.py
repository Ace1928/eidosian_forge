import numpy as np
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class UnaryStats(StatsCommand):
    """Unary statistical operations.

    See Also
    --------
    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`__ --
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`__

    Examples
    --------
    >>> import copy
    >>> from nipype.interfaces import niftyseg
    >>> unary = niftyseg.UnaryStats()
    >>> unary.inputs.in_file = 'im1.nii'

    >>> # Test v operation
    >>> unary_v = copy.deepcopy(unary)
    >>> unary_v.inputs.operation = 'v'
    >>> unary_v.cmdline
    'seg_stats im1.nii -v'
    >>> unary_v.run()  # doctest: +SKIP

    >>> # Test vl operation
    >>> unary_vl = copy.deepcopy(unary)
    >>> unary_vl.inputs.operation = 'vl'
    >>> unary_vl.cmdline
    'seg_stats im1.nii -vl'
    >>> unary_vl.run()  # doctest: +SKIP

    >>> # Test x operation
    >>> unary_x = copy.deepcopy(unary)
    >>> unary_x.inputs.operation = 'x'
    >>> unary_x.cmdline
    'seg_stats im1.nii -x'
    >>> unary_x.run()  # doctest: +SKIP

    """
    input_spec = UnaryStatsInput