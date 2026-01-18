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
class ComputeDVARS(BaseInterface):
    """
    Computes the DVARS.
    """
    input_spec = ComputeDVARSInputSpec
    output_spec = ComputeDVARSOutputSpec
    _references = [{'entry': BibTeX('@techreport{nichols_notes_2013,\n    address = {Coventry, UK},\n    title = {Notes on {Creating} a {Standardized} {Version} of {DVARS}},\n    url = {http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/scripts/fsl/standardizeddvars.pdf},\n    urldate = {2016-08-16},\n    institution = {University of Warwick},\n    author = {Nichols, Thomas},\n    year = {2013}\n}'), 'tags': ['method']}, {'entry': BibTeX('@article{power_spurious_2012,\n    title = {Spurious but systematic correlations in functional connectivity {MRI} networks arise from subject motion},\n    volume = {59},\n    doi = {10.1016/j.neuroimage.2011.10.018},\n    number = {3},\n    urldate = {2016-08-16},\n    journal = {NeuroImage},\n    author = {Power, Jonathan D. and Barnes, Kelly A. and Snyder, Abraham Z. and Schlaggar, Bradley L. and Petersen, Steven E.},\n    year = {2012},\n    pages = {2142--2154},\n}\n'), 'tags': ['method']}]

    def __init__(self, **inputs):
        self._results = {}
        super(ComputeDVARS, self).__init__(**inputs)

    def _gen_fname(self, suffix, ext=None):
        fname, in_ext = op.splitext(op.basename(self.inputs.in_file))
        if in_ext == '.gz':
            fname, in_ext2 = op.splitext(fname)
            in_ext = in_ext2 + in_ext
        if ext is None:
            ext = in_ext
        if ext.startswith('.'):
            ext = ext[1:]
        return op.abspath('{}_{}.{}'.format(fname, suffix, ext))

    def _run_interface(self, runtime):
        dvars = compute_dvars(self.inputs.in_file, self.inputs.in_mask, remove_zerovariance=self.inputs.remove_zerovariance, variance_tol=self.inputs.variance_tol, intensity_normalization=self.inputs.intensity_normalization)
        self._results['avg_std'], self._results['avg_nstd'], self._results['avg_vxstd'] = np.mean(dvars, axis=1).astype(float)
        tr = None
        if isdefined(self.inputs.series_tr):
            tr = self.inputs.series_tr
        if self.inputs.save_std:
            out_file = self._gen_fname('dvars_std', ext='tsv')
            np.savetxt(out_file, dvars[0], fmt=b'%0.6f')
            self._results['out_std'] = out_file
            if self.inputs.save_plot:
                self._results['fig_std'] = self._gen_fname('dvars_std', ext=self.inputs.figformat)
                fig = plot_confound(dvars[0], self.inputs.figsize, 'Standardized DVARS', series_tr=tr)
                fig.savefig(self._results['fig_std'], dpi=float(self.inputs.figdpi), format=self.inputs.figformat, bbox_inches='tight')
                fig.clf()
        if self.inputs.save_nstd:
            out_file = self._gen_fname('dvars_nstd', ext='tsv')
            np.savetxt(out_file, dvars[1], fmt=b'%0.6f')
            self._results['out_nstd'] = out_file
            if self.inputs.save_plot:
                self._results['fig_nstd'] = self._gen_fname('dvars_nstd', ext=self.inputs.figformat)
                fig = plot_confound(dvars[1], self.inputs.figsize, 'DVARS', series_tr=tr)
                fig.savefig(self._results['fig_nstd'], dpi=float(self.inputs.figdpi), format=self.inputs.figformat, bbox_inches='tight')
                fig.clf()
        if self.inputs.save_vxstd:
            out_file = self._gen_fname('dvars_vxstd', ext='tsv')
            np.savetxt(out_file, dvars[2], fmt=b'%0.6f')
            self._results['out_vxstd'] = out_file
            if self.inputs.save_plot:
                self._results['fig_vxstd'] = self._gen_fname('dvars_vxstd', ext=self.inputs.figformat)
                fig = plot_confound(dvars[2], self.inputs.figsize, 'Voxelwise std DVARS', series_tr=tr)
                fig.savefig(self._results['fig_vxstd'], dpi=float(self.inputs.figdpi), format=self.inputs.figformat, bbox_inches='tight')
                fig.clf()
        if self.inputs.save_all:
            out_file = self._gen_fname('dvars', ext='tsv')
            np.savetxt(out_file, np.vstack(dvars).T, fmt=b'%0.8f', delimiter=b'\t', header='std DVARS\tnon-std DVARS\tvx-wise std DVARS', comments='')
            self._results['out_all'] = out_file
        return runtime

    def _list_outputs(self):
        return self._results