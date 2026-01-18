import os
import re
from copy import deepcopy
import numpy as np
from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .spatialimages import HeaderDataError, ImageDataError, SpatialHeader, SpatialImage
from .volumeutils import Recoder
class AFNIHeader(SpatialHeader):
    """Class for AFNI header"""

    def __init__(self, info):
        """
        Initialize AFNI header object

        Parameters
        ----------
        info : dict
            Information from HEAD file as obtained by :func:`parse_AFNI_header`

        Examples
        --------
        >>> fname = os.path.join(datadir, 'example4d+orig.HEAD')
        >>> header = AFNIHeader(parse_AFNI_header(fname))
        >>> header.get_data_dtype().str
        '<i2'
        >>> header.get_zooms()
        (3.0, 3.0, 3.0, 3.0)
        >>> header.get_data_shape()
        (33, 41, 25, 3)
        """
        self.info = info
        dt = _get_datatype(self.info)
        super().__init__(data_dtype=dt, shape=self._calc_data_shape(), zooms=self._calc_zooms())

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            raise AFNIHeaderError('Cannot create AFNIHeader from nothing.')
        if type(header) == klass:
            return header.copy()
        raise AFNIHeaderError('Cannot create AFNIHeader from non-AFNIHeader.')

    @classmethod
    def from_fileobj(klass, fileobj):
        info = parse_AFNI_header(fileobj)
        return klass(info)

    def copy(self):
        return AFNIHeader(deepcopy(self.info))

    def _calc_data_shape(self):
        """
        Calculate the output shape of the image data

        Returns length 3 tuple for 3D image, length 4 tuple for 4D.

        Returns
        -------
        (x, y, z, t) : tuple of int

        Notes
        -----
        ``DATASET_RANK[0]`` gives number of spatial dimensions (and apparently
        must be 3). ``DATASET_RANK[1]`` gives the number of sub-bricks.
        ``DATASET_DIMENSIONS`` is length 3, giving the number of voxels in i,
        j, k.
        """
        dset_rank = self.info['DATASET_RANK']
        shape = tuple(self.info['DATASET_DIMENSIONS'][:dset_rank[0]])
        n_vols = dset_rank[1]
        return shape + (n_vols,)

    def _calc_zooms(self):
        """
        Get image zooms from header data

        Spatial axes are first three indices, time axis is last index. If
        dataset is not a time series the last value will be zero.

        Returns
        -------
        zooms : tuple

        Notes
        -----
        Gets zooms from attributes ``DELTA`` and ``TAXIS_FLOATS``.

        ``DELTA`` gives (x,y,z) voxel sizes.

        ``TAXIS_FLOATS`` should be length 5, with first entry giving "Time
        origin", and second giving "Time step (TR)".
        """
        xyz_step = tuple(np.abs(self.info['DELTA']))
        t_step = self.info.get('TAXIS_FLOATS', (0, 0))
        if len(t_step) > 0:
            t_step = (t_step[1],)
        return xyz_step + t_step

    def get_space(self):
        """
        Return label for anatomical space to which this dataset is aligned.

        Returns
        -------
        space : str
            AFNI "space" designation; one of [ORIG, ANAT, TLRC, MNI]

        Notes
        -----
        There appears to be documentation for these spaces at
        https://afni.nimh.nih.gov/pub/dist/atlases/elsedemo/AFNI_atlas_spaces.niml
        """
        listed_space = self.info.get('TEMPLATE_SPACE', 0)
        space = space_codes.space[listed_space]
        return space

    def get_affine(self):
        """
        Returns affine of dataset

        Examples
        --------
        >>> fname = os.path.join(datadir, 'example4d+orig.HEAD')
        >>> header = AFNIHeader(parse_AFNI_header(fname))
        >>> header.get_affine()
        array([[ -3.    ,  -0.    ,  -0.    ,  49.5   ],
               [ -0.    ,  -3.    ,  -0.    ,  82.312 ],
               [  0.    ,   0.    ,   3.    , -52.3511],
               [  0.    ,   0.    ,   0.    ,   1.    ]])
        """
        affine = np.asarray(self.info['IJK_TO_DICOM_REAL']).reshape(3, 4)
        affine = np.row_stack((affine * [[-1], [-1], [1]], [0, 0, 0, 1]))
        return affine

    def get_data_scaling(self):
        """
        AFNI applies volume-specific data scaling

        Examples
        --------
        >>> fname = os.path.join(datadir, 'scaled+tlrc.HEAD')
        >>> header = AFNIHeader(parse_AFNI_header(fname))
        >>> header.get_data_scaling()
        array([3.883363e-08])
        """
        floatfacs = self.info.get('BRICK_FLOAT_FACS', None)
        if floatfacs is None or not np.any(floatfacs):
            return None
        scale = np.ones(self.info['DATASET_RANK'][1])
        floatfacs = np.atleast_1d(floatfacs)
        scale[floatfacs.nonzero()] = floatfacs[floatfacs.nonzero()]
        return scale

    def get_slope_inter(self):
        """
        Use `self.get_data_scaling()` instead

        Holdover because ``AFNIArrayProxy`` (inheriting from ``ArrayProxy``)
        requires this functionality so as to not error.
        """
        return (None, None)

    def get_data_offset(self):
        """Data offset in BRIK file

        Offset is always 0.
        """
        return DATA_OFFSET

    def get_volume_labels(self):
        """
        Returns volume labels

        Returns
        -------
        labels : list of str
            Labels for volumes along fourth dimension

        Examples
        --------
        >>> header = AFNIHeader(parse_AFNI_header(os.path.join(datadir, 'example4d+orig.HEAD')))
        >>> header.get_volume_labels()
        ['#0', '#1', '#2']
        """
        labels = self.info.get('BRICK_LABS', None)
        if labels is not None:
            labels = labels.split('~')
        return labels