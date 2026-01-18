import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegAverage(NiftyRegCommand):
    """Interface for executable reg_average from NiftyReg platform.

    Compute average matrix or image from a list of matrices or image.
    The tool can be use to resample images given input transformation
    parametrisation as well as to demean transformations in Euclidean or
    log-Euclidean space.

    This interface is different than the others in the way that the options
    will be written in a command file that is given as a parameter.

    `Source code <https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg>`_

    Examples
    --------

    >>> from nipype.interfaces import niftyreg
    >>> node = niftyreg.RegAverage()
    >>> one_file = 'im1.nii'
    >>> two_file = 'im2.nii'
    >>> three_file = 'im3.nii'
    >>> node.inputs.avg_files = [one_file, two_file, three_file]
    >>> node.cmdline  # doctest: +ELLIPSIS
    'reg_average --cmd_file .../reg_average_cmd'
    """
    _cmd = get_custom_path('reg_average')
    input_spec = RegAverageInputSpec
    output_spec = RegAverageOutputSpec
    _suffix = 'avg_out'

    def _gen_filename(self, name):
        if name == 'out_file':
            if isdefined(self.inputs.avg_lts_files):
                return self._gen_fname(self._suffix, ext='.txt')
            elif isdefined(self.inputs.avg_files):
                _, _, _ext = split_filename(self.inputs.avg_files[0])
                if _ext not in ['.nii', '.nii.gz', '.hdr', '.img', '.img.gz']:
                    return self._gen_fname(self._suffix, ext=_ext)
            return self._gen_fname(self._suffix, ext='.nii.gz')
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs['out_file'] = self.inputs.out_file
        else:
            outputs['out_file'] = self._gen_filename('out_file')
        return outputs

    @property
    def cmdline(self):
        """Rewrite the cmdline to write options in text_file."""
        argv = super(RegAverage, self).cmdline
        reg_average_cmd = os.path.join(os.getcwd(), 'reg_average_cmd')
        with open(reg_average_cmd, 'w') as f:
            f.write(argv)
        return '%s --cmd_file %s' % (self.cmd, reg_average_cmd)