import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from locale import getpreferredencoding
import numpy as np
from .affines import apply_affine, dot_reduce, from_matvec
from .eulerangles import euler2mat
from .fileslice import fileslice, strided_scalar
from .nifti1 import unit_codes
from .openers import ImageOpener
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import Recoder, array_from_file
def _calc_zooms(self):
    """Compute image zooms from header data.

        Spatial axis are first three.

        Returns
        -------
        zooms : array
            Length 3 array for 3D image, length 4 array for 4D image.

        Notes
        -----
        This routine gets called in ``__init__``, so may not be able to use
        some attributes available in the fully initialized object.
        """
    slice_gap = self._get_unique_image_prop('slice gap')
    n_dim = 4 if self._get_n_vols() > 1 else 3
    zooms = np.ones(n_dim)
    zooms[:2] = self._get_unique_image_prop('pixel spacing')
    slice_thickness = self._get_unique_image_prop('slice thickness')
    zooms[2] = slice_thickness + slice_gap
    if len(zooms) > 3 and self.general_info['dyn_scan']:
        if len(self.general_info['repetition_time']) > 1:
            warnings.warn('multiple TRs found in .PAR file')
        zooms[3] = self.general_info['repetition_time'][0] / 1000.0
    return zooms