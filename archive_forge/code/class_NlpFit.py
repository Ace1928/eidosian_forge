import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class NlpFit(CommandLine):
    """Hierarchial non-linear fitting with bluring.

    This tool is part of the minc-widgets package:

    https://github.com/BIC-MNI/minc-widgets/blob/master/nlpfit/nlpfit

    Examples
    --------

    >>> from nipype.interfaces.minc import NlpFit
    >>> from nipype.interfaces.minc.testdata import nonempty_minc_data, nlp_config
    >>> from nipype.testing import example_data

    >>> source = nonempty_minc_data(0)
    >>> target = nonempty_minc_data(1)
    >>> source_mask = nonempty_minc_data(2)
    >>> config = nlp_config
    >>> initial = example_data('minc_initial.xfm')
    >>> nlpfit = NlpFit(config_file=config, init_xfm=initial, source_mask=source_mask, source=source, target=target)
    >>> nlpfit.run() # doctest: +SKIP
    """
    input_spec = NlpFitInputSpec
    output_spec = NlpFitOutputSpec
    _cmd = 'nlpfit'

    def _gen_filename(self, name):
        if name == 'output_xfm':
            output_xfm = self.inputs.output_xfm
            if isdefined(output_xfm):
                return os.path.abspath(output_xfm)
            else:
                return aggregate_filename([self.inputs.source, self.inputs.target], 'nlpfit_xfm_output') + '.xfm'
        else:
            raise NotImplemented

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_xfm'] = os.path.abspath(self._gen_filename('output_xfm'))
        assert os.path.exists(outputs['output_xfm'])
        if 'grid' in open(outputs['output_xfm'], 'r').read():
            outputs['output_grid'] = re.sub('.(nlxfm|xfm)$', '_grid_0.mnc', outputs['output_xfm'])
        return outputs