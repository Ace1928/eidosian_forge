import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class CalculateMedian(BaseInterface):
    """
    Computes an average of the median across one or more 4D Nifti timeseries

    Example
    -------
    >>> from nipype.algorithms.misc import CalculateMedian
    >>> mean = CalculateMedian()
    >>> mean.inputs.in_files = 'functional.nii'
    >>> mean.run() # doctest: +SKIP

    """
    input_spec = CalculateMedianInputSpec
    output_spec = CalculateMedianOutputSpec

    def __init__(self, *args, **kwargs):
        super(CalculateMedian, self).__init__(*args, **kwargs)
        self._median_files = []

    def _gen_fname(self, suffix, idx=None, ext=None):
        if idx:
            in_file = self.inputs.in_files[idx]
        elif isinstance(self.inputs.in_files, list):
            in_file = self.inputs.in_files[0]
        else:
            in_file = self.inputs.in_files
        fname, in_ext = op.splitext(op.basename(in_file))
        if in_ext == '.gz':
            fname, in_ext2 = op.splitext(fname)
            in_ext = in_ext2 + in_ext
        if ext is None:
            ext = in_ext
        if ext.startswith('.'):
            ext = ext[1:]
        if self.inputs.median_file:
            outname = self.inputs.median_file
        else:
            outname = '{}_{}'.format(fname, suffix)
        if idx:
            outname += str(idx)
        return op.abspath('{}.{}'.format(outname, ext))

    def _run_interface(self, runtime):
        total = None
        self._median_files = []
        for idx, fname in enumerate(ensure_list(self.inputs.in_files)):
            img = nb.load(fname)
            data = np.median(img.get_fdata(), axis=3)
            if self.inputs.median_per_file:
                self._median_files.append(self._write_nifti(img, data, idx))
            elif total is None:
                total = data
            else:
                total += data
        if not self.inputs.median_per_file:
            self._median_files.append(self._write_nifti(img, total, idx))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['median_files'] = self._median_files
        return outputs

    def _write_nifti(self, img, data, idx, suffix='median'):
        if self.inputs.median_per_file:
            median_img = nb.Nifti1Image(data, img.affine, img.header)
            filename = self._gen_fname(suffix, idx=idx)
        else:
            median_img = nb.Nifti1Image(data / (idx + 1), img.affine, img.header)
            filename = self._gen_fname(suffix)
        median_img.to_filename(filename)
        return filename