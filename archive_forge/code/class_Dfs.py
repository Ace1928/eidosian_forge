import os
import re as regex
from ..base import (
class Dfs(CommandLine):
    """
    Surface Generator
    Generates mesh surfaces using an isosurface algorithm.

    http://brainsuite.org/processing/surfaceextraction/inner-cortical-surface/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> from nipype.testing import example_data
    >>> dfs = brainsuite.Dfs()
    >>> dfs.inputs.inputVolumeFile = example_data('structural.nii')
    >>> results = dfs.run() #doctest: +SKIP

    """
    input_spec = DfsInputSpec
    output_spec = DfsOutputSpec
    _cmd = 'dfs'

    def _format_arg(self, name, spec, value):
        if name == 'tessellationThreshold':
            return ''
        if name == 'specialTessellation':
            threshold = self.inputs.tessellationThreshold
            return spec.argstr % {'greater_than': ''.join('-gt %f' % threshold), 'less_than': ''.join('-lt %f' % threshold), 'equal_to': ''.join('-eq %f' % threshold)}[value]
        return super(Dfs, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):
        inputs = self.inputs.get()
        if isdefined(inputs[name]):
            return os.path.abspath(inputs[name])
        if name == 'outputSurfaceFile':
            return getFileName(self.inputs.inputVolumeFile, '.inner.cortex.dfs')
        return None

    def _list_outputs(self):
        return l_outputs(self)