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
class TCompCor(CompCor):
    """
    Interface for tCompCor. Computes a ROI mask based on variance of voxels.

    Example
    -------
    >>> ccinterface = TCompCor()
    >>> ccinterface.inputs.realigned_file = 'functional.nii'
    >>> ccinterface.inputs.mask_files = 'mask.nii'
    >>> ccinterface.inputs.num_components = 1
    >>> ccinterface.inputs.pre_filter = 'polynomial'
    >>> ccinterface.inputs.regress_poly_degree = 2
    >>> ccinterface.inputs.percentile_threshold = .03

    """
    input_spec = TCompCorInputSpec
    output_spec = TCompCorOutputSpec

    def __init__(self, *args, **kwargs):
        """exactly the same as compcor except the header"""
        super(TCompCor, self).__init__(*args, **kwargs)
        self._header = 'tCompCor'
        self._mask_files = []

    def _process_masks(self, mask_images, timeseries=None):
        out_images = []
        self._mask_files = []
        timeseries = np.asanyarray(timeseries)
        for i, img in enumerate(mask_images):
            mask = np.asanyarray(img.dataobj).astype(bool)
            imgseries = timeseries[mask, :]
            imgseries = regress_poly(2, imgseries)[0]
            tSTD = _compute_tSTD(imgseries, 0, axis=-1)
            threshold_std = np.percentile(tSTD, np.round(100.0 * (1.0 - self.inputs.percentile_threshold)).astype(int))
            mask_data = np.zeros_like(mask)
            mask_data[mask != 0] = tSTD >= threshold_std
            out_image = nb.Nifti1Image(mask_data, affine=img.affine, header=img.header)
            mask_file = os.path.abspath('mask_{:03d}.nii.gz'.format(i))
            out_image.to_filename(mask_file)
            IFLOGGER.debug('tCompcor computed and saved mask of shape %s to mask_file %s', str(mask.shape), mask_file)
            self._mask_files.append(mask_file)
            out_images.append(out_image)
        return out_images

    def _list_outputs(self):
        outputs = super(TCompCor, self)._list_outputs()
        outputs['high_variance_masks'] = self._mask_files
        return outputs