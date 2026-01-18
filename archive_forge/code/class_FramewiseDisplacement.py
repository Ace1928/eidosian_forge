import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
class FramewiseDisplacement(BaseInterface):
    """
    Calculate the :abbr:`FD (framewise displacement)` as in [Power2012]_.
    This implementation reproduces the calculation in fsl_motion_outliers

    .. [Power2012] Power et al., Spurious but systematic correlations in functional
         connectivity MRI networks arise from subject motion, NeuroImage 59(3),
         2012. doi:`10.1016/j.neuroimage.2011.10.018
         <https://doi.org/10.1016/j.neuroimage.2011.10.018>`_.


    """
    input_spec = FramewiseDisplacementInputSpec
    output_spec = FramewiseDisplacementOutputSpec
    _references = [{'entry': BibTeX('@article{power_spurious_2012,\n    title = {Spurious but systematic correlations in functional connectivity {MRI} networks arise from subject motion},\n    volume = {59},\n    doi = {10.1016/j.neuroimage.2011.10.018},\n    number = {3},\n    urldate = {2016-08-16},\n    journal = {NeuroImage},\n    author = {Power, Jonathan D. and Barnes, Kelly A. and Snyder, Abraham Z. and Schlaggar, Bradley L. and Petersen, Steven E.},\n    year = {2012},\n    pages = {2142--2154},\n}\n'), 'tags': ['method']}]

    def _run_interface(self, runtime):
        mpars = np.loadtxt(self.inputs.in_file)
        mpars = np.apply_along_axis(func1d=normalize_mc_params, axis=1, arr=mpars, source=self.inputs.parameter_source)
        diff = mpars[:-1, :6] - mpars[1:, :6]
        diff[:, 3:6] *= self.inputs.radius
        fd_res = np.abs(diff).sum(axis=1)
        self._results = {'out_file': op.abspath(self.inputs.out_file), 'fd_average': float(fd_res.mean())}
        np.savetxt(self.inputs.out_file, fd_res, header='FramewiseDisplacement', comments='')
        if self.inputs.save_plot:
            tr = None
            if isdefined(self.inputs.series_tr):
                tr = self.inputs.series_tr
            if self.inputs.normalize and tr is None:
                IFLOGGER.warning('FD plot cannot be normalized if TR is not set')
            self._results['out_figure'] = op.abspath(self.inputs.out_figure)
            fig = plot_confound(fd_res, self.inputs.figsize, 'FD', units='mm', series_tr=tr, normalize=self.inputs.normalize)
            fig.savefig(self._results['out_figure'], dpi=float(self.inputs.figdpi), format=self.inputs.out_figure[-3:], bbox_inches='tight')
            fig.clf()
        return runtime

    def _list_outputs(self):
        return self._results