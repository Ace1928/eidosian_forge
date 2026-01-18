import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class PlotTimeSeries(FSLCommand):
    """Use fsl_tsplot to create images of time course plots.

    Examples
    --------

    >>> import nipype.interfaces.fsl as fsl
    >>> plotter = fsl.PlotTimeSeries()
    >>> plotter.inputs.in_file = 'functional.par'
    >>> plotter.inputs.title = 'Functional timeseries'
    >>> plotter.inputs.labels = ['run1', 'run2']
    >>> plotter.run() #doctest: +SKIP


    """
    _cmd = 'fsl_tsplot'
    input_spec = PlotTimeSeriesInputSpec
    output_spec = PlotTimeSeriesOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'in_file':
            if isinstance(value, list):
                args = ','.join(value)
                return '-i %s' % args
            else:
                return '-i %s' % value
        elif name == 'labels':
            if isinstance(value, list):
                args = ','.join(value)
                return '-a %s' % args
            else:
                return '-a %s' % value
        elif name == 'title':
            return "-t '%s'" % value
        elif name == 'plot_range':
            return '--start=%d --finish=%d' % value
        elif name == 'y_range':
            return '--ymin=%d --ymax=%d' % value
        elif name == 'plot_size':
            return '-h %d -w %d' % value
        return super(PlotTimeSeries, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_file = self.inputs.out_file
        if not isdefined(out_file):
            if isinstance(self.inputs.in_file, list):
                infile = self.inputs.in_file[0]
            else:
                infile = self.inputs.in_file
            out_file = self._gen_fname(infile, ext='.png')
        outputs['out_file'] = os.path.abspath(out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()['out_file']
        return None