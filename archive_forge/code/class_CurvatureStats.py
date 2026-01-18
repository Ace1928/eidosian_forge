import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class CurvatureStats(FSCommand):
    """
    In its simplest usage, 'mris_curvature_stats' will compute a set
    of statistics on its input <curvFile>. These statistics are the
    mean and standard deviation of the particular curvature on the
    surface, as well as the results from several surface-based
    integrals.

    Additionally, 'mris_curvature_stats' can report the max/min
    curvature values, and compute a simple histogram based on
    all curvature values.

    Curvatures can also be normalised and constrained to a given
    range before computation.

    Principal curvature (K, H, k1 and k2) calculations on a surface
    structure can also be performed, as well as several functions
    derived from k1 and k2.

    Finally, all output to the console, as well as any new
    curvatures that result from the above calculations can be
    saved to a series of text and binary-curvature files.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import CurvatureStats
    >>> curvstats = CurvatureStats()
    >>> curvstats.inputs.hemisphere = 'lh'
    >>> curvstats.inputs.curvfile1 = 'lh.pial'
    >>> curvstats.inputs.curvfile2 = 'lh.pial'
    >>> curvstats.inputs.surface = 'lh.pial'
    >>> curvstats.inputs.out_file = 'lh.curv.stats'
    >>> curvstats.inputs.values = True
    >>> curvstats.inputs.min_max = True
    >>> curvstats.inputs.write = True
    >>> curvstats.cmdline
    'mris_curvature_stats -m -o lh.curv.stats -F pial -G --writeCurvatureFiles subject_id lh pial pial'
    """
    _cmd = 'mris_curvature_stats'
    input_spec = CurvatureStatsInputSpec
    output_spec = CurvatureStatsOutputSpec

    def _format_arg(self, name, spec, value):
        if name in ['surface', 'curvfile1', 'curvfile2']:
            prefix = os.path.basename(value).split('.')[1]
            return spec.argstr % prefix
        return super(CurvatureStats, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            copy2subjdir(self, self.inputs.surface, 'surf')
            copy2subjdir(self, self.inputs.curvfile1, 'surf')
            copy2subjdir(self, self.inputs.curvfile2, 'surf')
        return super(CurvatureStats, self).run(**inputs)