import os
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import split_filename
class BinaryMaths(MathsCommand):
    """Binary mathematical operations.

    See Also
    --------
    `Source code <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg>`__ --
    `Documentation <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_documentation>`__

    Examples
    --------
    >>> import copy
    >>> from nipype.interfaces import niftyseg
    >>> binary = niftyseg.BinaryMaths()
    >>> binary.inputs.in_file = 'im1.nii'
    >>> binary.inputs.output_datatype = 'float'

    >>> # Test sub operation
    >>> binary_sub = copy.deepcopy(binary)
    >>> binary_sub.inputs.operation = 'sub'
    >>> binary_sub.inputs.operand_file = 'im2.nii'
    >>> binary_sub.cmdline
    'seg_maths im1.nii -sub im2.nii -odt float im1_sub.nii'
    >>> binary_sub.run()  # doctest: +SKIP

    >>> # Test mul operation
    >>> binary_mul = copy.deepcopy(binary)
    >>> binary_mul.inputs.operation = 'mul'
    >>> binary_mul.inputs.operand_value = 2.0
    >>> binary_mul.cmdline
    'seg_maths im1.nii -mul 2.00000000 -odt float im1_mul.nii'
    >>> binary_mul.run()  # doctest: +SKIP

    >>> # Test llsnorm operation
    >>> binary_llsnorm = copy.deepcopy(binary)
    >>> binary_llsnorm.inputs.operation = 'llsnorm'
    >>> binary_llsnorm.inputs.operand_file = 'im2.nii'
    >>> binary_llsnorm.cmdline
    'seg_maths im1.nii -llsnorm im2.nii -odt float im1_llsnorm.nii'
    >>> binary_llsnorm.run()  # doctest: +SKIP

    >>> # Test splitinter operation
    >>> binary_splitinter = copy.deepcopy(binary)
    >>> binary_splitinter.inputs.operation = 'splitinter'
    >>> binary_splitinter.inputs.operand_str = 'z'
    >>> binary_splitinter.cmdline
    'seg_maths im1.nii -splitinter z -odt float im1_splitinter.nii'
    >>> binary_splitinter.run()  # doctest: +SKIP

    """
    input_spec = BinaryMathsInput

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for seg_maths."""
        if opt == 'operand_str' and self.inputs.operation != 'splitinter':
            err = 'operand_str set but with an operation different than "splitinter"'
            raise NipypeInterfaceError(err)
        if opt == 'operation':
            if val in ['pow', 'thr', 'uthr', 'smo', 'edge', 'sobel3', 'sobel5', 'smol']:
                if not isdefined(self.inputs.operand_value):
                    err = 'operand_value not set for {0}.'.format(val)
                    raise NipypeInterfaceError(err)
            elif val in ['min', 'llsnorm', 'masknan', 'hdr_copy']:
                if not isdefined(self.inputs.operand_file):
                    err = 'operand_file not set for {0}.'.format(val)
                    raise NipypeInterfaceError(err)
            elif val == 'splitinter':
                if not isdefined(self.inputs.operand_str):
                    err = 'operand_str not set for splitinter.'
                    raise NipypeInterfaceError(err)
        if opt == 'operand_value' and float(val) == 0.0:
            return '0'
        return super(BinaryMaths, self)._format_arg(opt, spec, val)

    def _overload_extension(self, value, name=None):
        if self.inputs.operation == 'hdr_copy':
            path, base, _ = split_filename(value)
            _, base, ext = split_filename(self.inputs.operand_file)
            suffix = self.inputs.operation
            return os.path.join(path, '{0}{1}{2}'.format(base, suffix, ext))
        else:
            return super(BinaryMaths, self)._overload_extension(value, name)