import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class XfmInvert(CommandLine):
    """Invert an xfm transform file.

    Examples
    --------

    >>> from nipype.interfaces.minc import XfmAvg
    >>> from nipype.testing import example_data

    >>> xfm = example_data('minc_initial.xfm')
    >>> invert = XfmInvert(input_file=xfm)
    >>> invert.run() # doctest: +SKIP
    """
    input_spec = XfmInvertInputSpec
    output_spec = XfmInvertOutputSpec
    _cmd = 'xfminvert'

    def _gen_filename(self, name):
        if name == 'output_file':
            output_file = self.inputs.output_file
            if isdefined(output_file):
                return os.path.abspath(output_file)
            else:
                return aggregate_filename([self.inputs.input_file], 'xfminvert_output') + '.xfm'
        else:
            raise NotImplemented

    def _gen_outfilename(self):
        return self._gen_filename('output_file')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_file'] = os.path.abspath(self._gen_outfilename())
        assert os.path.exists(outputs['output_file'])
        if 'grid' in open(outputs['output_file'], 'r').read():
            outputs['output_grid'] = re.sub('.(nlxfm|xfm)$', '_grid_0.mnc', outputs['output_file'])
        return outputs