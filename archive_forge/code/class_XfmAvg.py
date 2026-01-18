import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class XfmAvg(CommandLine):
    """Average a number of xfm transforms using matrix logs and exponents.
    The program xfmavg calls Octave for numerical work.

    This tool is part of the minc-widgets package:

    https://github.com/BIC-MNI/minc-widgets/tree/master/xfmavg

    Examples
    --------

    >>> from nipype.interfaces.minc import XfmAvg
    >>> from nipype.interfaces.minc.testdata import nonempty_minc_data, nlp_config
    >>> from nipype.testing import example_data

    >>> xfm1 = example_data('minc_initial.xfm')
    >>> xfm2 = example_data('minc_initial.xfm')  # cheating for doctest
    >>> xfmavg = XfmAvg(input_files=[xfm1, xfm2])
    >>> xfmavg.run() # doctest: +SKIP
    """
    input_spec = XfmAvgInputSpec
    output_spec = XfmAvgOutputSpec
    _cmd = 'xfmavg'

    def _gen_filename(self, name):
        if name == 'output_file':
            output_file = self.inputs.output_file
            if isdefined(output_file):
                return os.path.abspath(output_file)
            else:
                return aggregate_filename(self.inputs.input_files, 'xfmavg_output') + '.xfm'
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