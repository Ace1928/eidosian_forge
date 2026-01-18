import warnings
from io import BytesIO
import numpy as np
from . import analyze  # module import
from .batteryrunners import Report
from .optpkg import optional_package
from .spatialimages import HeaderDataError, HeaderTypeError
class Spm99AnalyzeHeader(SpmAnalyzeHeader):
    """Class for SPM99 variant of basic Analyze header

    SPM99 variant adds the following to basic Analyze format:

    * voxel origin;
    * slope scaling of data.
    """

    def get_origin_affine(self):
        """Get affine from header, using SPM origin field if sensible

        The default translations are got from the ``origin``
        field, if set, or from the center of the image otherwise.

        Examples
        --------
        >>> hdr = Spm99AnalyzeHeader()
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.set_zooms((3, 2, 1))
        >>> hdr.default_x_flip
        True
        >>> hdr.get_origin_affine() # from center of image
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        >>> hdr['origin'][:3] = [3,4,5]
        >>> hdr.get_origin_affine() # using origin
        array([[-3.,  0.,  0.,  6.],
               [ 0.,  2.,  0., -6.],
               [ 0.,  0.,  1., -4.],
               [ 0.,  0.,  0.,  1.]])
        >>> hdr['origin'] = 0 # unset origin
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.get_origin_affine() # from center of image
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        """
        hdr = self._structarr
        zooms = hdr['pixdim'][1:4].copy()
        if self.default_x_flip:
            zooms[0] *= -1
        origin = hdr['origin'][:3]
        dims = hdr['dim'][1:4]
        if np.any(origin) and np.all(origin > -dims) and np.all(origin < dims * 2):
            origin = origin - 1
        else:
            origin = (dims - 1) / 2.0
        aff = np.eye(4)
        aff[:3, :3] = np.diag(zooms)
        aff[:3, -1] = -origin * zooms
        return aff
    get_best_affine = get_origin_affine

    def set_origin_from_affine(self, affine):
        """Set SPM origin to header from affine matrix.

        The ``origin`` field was read but not written by SPM99 and 2.  It was
        used for storing a central voxel coordinate, that could be used in
        aligning the image to some standard position - a proxy for a full
        translation vector that was usually stored in a separate matlab .mat
        file.

        Nifti uses the space occupied by the SPM ``origin`` field for important
        other information (the transform codes), so writing the origin will
        make the header a confusing Nifti file.  If you work with both Analyze
        and Nifti, you should probably avoid doing this.

        Parameters
        ----------
        affine : array-like, shape (4,4)
           Affine matrix to set

        Returns
        -------
        None

        Examples
        --------
        >>> hdr = Spm99AnalyzeHeader()
        >>> hdr.set_data_shape((3, 5, 7))
        >>> hdr.set_zooms((3,2,1))
        >>> hdr.get_origin_affine()
        array([[-3.,  0.,  0.,  3.],
               [ 0.,  2.,  0., -4.],
               [ 0.,  0.,  1., -3.],
               [ 0.,  0.,  0.,  1.]])
        >>> affine = np.diag([3,2,1,1])
        >>> affine[:3,3] = [-6, -6, -4]
        >>> hdr.set_origin_from_affine(affine)
        >>> np.all(hdr['origin'][:3] == [3,4,5])
        True
        >>> hdr.get_origin_affine()
        array([[-3.,  0.,  0.,  6.],
               [ 0.,  2.,  0., -6.],
               [ 0.,  0.,  1., -4.],
               [ 0.,  0.,  0.,  1.]])
        """
        if affine.shape != (4, 4):
            raise ValueError('Need 4x4 affine to set')
        hdr = self._structarr
        RZS = affine[:3, :3]
        Z = np.sqrt(np.sum(RZS * RZS, axis=0))
        T = affine[:3, 3]
        hdr['origin'][:3] = -T / Z + 1

    @classmethod
    def _get_checks(klass):
        checks = super()._get_checks()
        return checks + (klass._chk_origin,)

    @staticmethod
    def _chk_origin(hdr, fix=False):
        rep = Report(HeaderDataError)
        origin = hdr['origin'][0:3]
        dims = hdr['dim'][1:4]
        if not np.any(origin) or (np.all(origin > -dims) and np.all(origin < dims * 2)):
            return (hdr, rep)
        rep.problem_level = 20
        rep.problem_msg = 'very large origin values relative to dims'
        if fix:
            rep.fix_msg = 'leaving as set, ignoring for affine'
        return (hdr, rep)