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
class PARRECHeader(SpatialHeader):
    """PAR/REC header"""

    def __init__(self, info, image_defs, permit_truncated=False, strict_sort=False):
        """
        Parameters
        ----------
        info : dict
            "General information" from the PAR file (as returned by
            `parse_PAR_header()`).
        image_defs : array
            Structured array with image definitions from the PAR file (as
            returned by `parse_PAR_header()`).
        permit_truncated : bool, optional
            If True, a warning is emitted instead of an error when a truncated
            recording is detected.
        strict_sort : bool, optional, keyword-only
            If True, a larger number of header fields are used while sorting
            the REC data array.  This may produce a different sort order than
            `strict_sort=False`, where volumes are sorted by the order in which
            the slices appear in the .PAR file.
        """
        self.general_info = info.copy()
        self.image_defs = image_defs.copy()
        self.permit_truncated = permit_truncated
        self.strict_sort = strict_sort
        _truncation_checks(info, image_defs, permit_truncated)
        bitpix = self._get_unique_image_prop('image pixel size')
        if bitpix not in (8, 16):
            raise PARRECError(f'Only 8- and 16-bit data supported (not {bitpix}) please report this to the nibabel developers')
        dt = np.dtype('uint' + str(bitpix)).newbyteorder('<')
        super().__init__(data_dtype=dt, shape=self._calc_data_shape(), zooms=self._calc_zooms())

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            raise PARRECError('Cannot create PARRECHeader from air.')
        if type(header) == klass:
            return header.copy()
        raise PARRECError('Cannot create PARREC header from non-PARREC header.')

    @classmethod
    def from_fileobj(klass, fileobj, permit_truncated=False, strict_sort=False):
        info, image_defs = parse_PAR_header(fileobj)
        return klass(info, image_defs, permit_truncated, strict_sort)

    def copy(self):
        return PARRECHeader(deepcopy(self.general_info), self.image_defs.copy(), self.permit_truncated, self.strict_sort)

    def as_analyze_map(self):
        """Convert PAR parameters to NIFTI1 format"""
        descr = f'{self.general_info['exam_name']};{self.general_info['patient_name']};{self.general_info['exam_date'].replace(' ', '')};{self.general_info['protocol_name']}'[:80]
        is_fmri = self.general_info['max_dynamics'] > 1
        t = 'sec' if is_fmri else 'unknown'
        xyzt_units = unit_codes['mm'] + unit_codes[t]
        return dict(descr=descr, xyzt_units=xyzt_units)

    def get_water_fat_shift(self):
        """Water fat shift, in pixels"""
        return self.general_info['water_fat_shift']

    def get_echo_train_length(self):
        """Echo train length of the recording"""
        return self.general_info['epi_factor']

    def get_q_vectors(self):
        """Get Q vectors from the data

        Returns
        -------
        q_vectors : None or array
            Array of q vectors (bvals * bvecs), or None if not a diffusion
            acquisition.
        """
        bvals, bvecs = self.get_bvals_bvecs()
        if bvals is None or bvecs is None:
            return None
        return bvecs * bvals[:, np.newaxis]

    def get_bvals_bvecs(self):
        """Get bvals and bvecs from data

        Returns
        -------
        b_vals : None or array
            Array of b values, shape (n_directions,), or None if not a
            diffusion acquisition.
        b_vectors : None or array
            Array of b vectors, shape (n_directions, 3), or None if not a
            diffusion acquisition.
        """
        if self.general_info['diffusion'] == 0:
            return (None, None)
        reorder = self.get_sorted_slice_indices()
        if len(self.get_data_shape()) == 3:
            return (None, None)
        else:
            n_slices, n_vols = self.get_data_shape()[-2:]
        bvals = self.image_defs['diffusion_b_factor'][reorder].reshape((n_slices, n_vols), order='F')
        assert not np.any(np.diff(bvals, axis=0))
        bvals = bvals[0]
        if 'diffusion' not in self.image_defs.dtype.names:
            return (bvals, None)
        bvecs = self.image_defs['diffusion'][reorder].reshape((n_slices, n_vols, 3), order='F')
        assert not np.any(np.diff(bvecs, axis=0))
        bvecs = bvecs[0]
        permute_to_psl = ACQ_TO_PSL[self.get_slice_orientation()]
        bvecs = apply_affine(np.linalg.inv(permute_to_psl), bvecs)
        return (bvals, bvecs)

    def get_def(self, name):
        """Return a single image definition field (or None if missing)"""
        idef = self.image_defs
        return idef[name] if name in idef.dtype.names else None

    def _get_unique_image_prop(self, name):
        """Scan image definitions and return unique value of a property.

        * Get array for named field of ``self.image_defs``;
        * Check that all rows in the array are the same and raise error
          otherwise;
        * Return the row.

        Parameters
        ----------
        name : str
            Name of the property in ``self.image_defs``

        Returns
        -------
        unique_value : scalar or array

        Raises
        ------
        PARRECError
            if the rows of ``self.image_defs[name]`` do not all compare equal.
        """
        props = self.image_defs[name]
        if np.any(np.diff(props, axis=0)):
            raise PARRECError(f'Varying {name} in image sequence ({props}). This is not supported.')
        return props[0]

    def get_data_offset(self):
        """PAR header always has 0 data offset (into REC file)"""
        return 0

    def set_data_offset(self, offset):
        """PAR header always has 0 data offset (into REC file)"""
        if offset != 0:
            raise PARRECError('PAR header assumes offset 0')

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

    def get_affine(self, origin='scanner'):
        """Compute affine transformation into scanner space.

        The method only considers global rotation and offset settings in the
        header and ignores potentially deviating information in the image
        definitions.

        Parameters
        ----------
        origin : {'scanner', 'fov'}
            Transformation origin. By default the transformation is computed
            relative to the scanner's iso center. If 'fov' is requested the
            transformation origin will be the center of the field of view
            instead.

        Returns
        -------
        aff : (4, 4) array
            4x4 array, with output axis order corresponding to RAS or (x,y,z)
            or (lr, pa, fh).

        Notes
        -----
        Transformations appear to be specified in (ap, fh, rl) axes.  The
        orientation of data is recorded in the "slice orientation" field of the
        PAR header "General Information".

        We need to:

        * translate to coordinates in terms of the center of the FOV
        * apply voxel size scaling
        * reorder / flip the data to Philips' PSL axes
        * apply the rotations
        * apply any isocenter scaling offset if `origin` == "scanner"
        * reorder and flip to RAS axes
        """
        ijk_shape = np.array(self.get_data_shape()[:3])
        to_center = from_matvec(np.eye(3), -(ijk_shape - 1) / 2.0)
        zoomer = np.diag(list(self.get_zooms()[:3]) + [1])
        slice_orientation = self.get_slice_orientation()
        permute_to_psl = ACQ_TO_PSL.get(slice_orientation)
        if permute_to_psl is None:
            raise PARRECError(f'Unknown slice orientation ({slice_orientation}).')
        ap_rot, fh_rot, rl_rot = self.general_info['angulation'] * DEG2RAD
        Mx = euler2mat(x=ap_rot)
        My = euler2mat(y=fh_rot)
        Mz = euler2mat(z=rl_rot)
        rot = from_matvec(dot_reduce(Mz, Mx, My))
        psl_aff = dot_reduce(rot, permute_to_psl, zoomer, to_center)
        if origin == 'scanner':
            iso_offset = self.general_info['off_center']
            psl_aff[:3, 3] += iso_offset
        return np.dot(PSL_TO_RAS, psl_aff)

    def _get_n_slices(self):
        """Get number of slices for output data"""
        return len(set(self.image_defs['slice number']))

    def _get_n_vols(self):
        """Get number of volumes for output data"""
        slice_nos = self.image_defs['slice number']
        vol_nos = vol_numbers(slice_nos)
        is_full = vol_is_full(slice_nos, self.general_info['max_slices'])
        return len(set(np.array(vol_nos)[is_full]))

    def _calc_data_shape(self):
        """Calculate the output shape of the image data

        Returns length 3 tuple for 3D image, length 4 tuple for 4D.

        Returns
        -------
        n_inplaneX : int
            number of voxels in X direction.
        n_inplaneY : int
            number of voxels in Y direction.
        n_slices : int
            number of slices.
        n_vols : int
            number of volumes or absent for 3D image.

        Notes
        -----
        This routine gets called in ``__init__``, so may not be able to use
        some attributes available in the fully initialized object.
        """
        inplane_shape = tuple(self._get_unique_image_prop('recon resolution'))
        shape = inplane_shape + (self._get_n_slices(),)
        n_vols = self._get_n_vols()
        return shape + (n_vols,) if n_vols > 1 else shape

    def get_data_scaling(self, method='dv'):
        """Returns scaling slope and intercept.

        Parameters
        ----------
        method : {'fp', 'dv'}
          Scaling settings to be reported -- see notes below.

        Returns
        -------
        slope : array
            scaling slope
        intercept : array
            scaling intercept

        Notes
        -----
        The PAR header contains two different scaling settings: 'dv' (value on
        console) and 'fp' (floating point value). Here is how they are defined:

        DV = PV * RS + RI
        FP = DV / (RS * SS)

        where:

        PV: value in REC
        RS: rescale slope
        RI: rescale intercept
        SS: scale slope
        """
        scale_slope = self.image_defs['scale slope']
        rescale_slope = self.image_defs['rescale slope']
        rescale_intercept = self.image_defs['rescale intercept']
        if method == 'dv':
            slope, intercept = (rescale_slope, rescale_intercept)
        elif method == 'fp':
            slope = 1.0 / scale_slope
            intercept = rescale_intercept / (rescale_slope * scale_slope)
        else:
            raise ValueError(f"Unknown scaling method '{method}'.")
        reorder = self.get_sorted_slice_indices()
        slope = slope[reorder]
        intercept = intercept[reorder]
        shape = (1, 1) + self.get_data_shape()[2:]
        slope = slope.reshape(shape, order='F')
        intercept = intercept.reshape(shape, order='F')
        return (slope, intercept)

    def get_slice_orientation(self):
        """Returns the slice orientation label.

        Returns
        -------
        orientation : {'transverse', 'sagittal', 'coronal'}
        """
        lab = self._get_unique_image_prop('slice orientation')
        return slice_orientation_codes.label[lab]

    def get_rec_shape(self):
        inplane_shape = tuple(self._get_unique_image_prop('recon resolution'))
        return inplane_shape + (len(self.image_defs),)

    def _strict_sort_order(self):
        """Determine the sort order based on several image definition fields.

        The fields taken into consideration, if present, are (in order from
        slowest to fastest variation after sorting):

            - image_defs['image_type_mr']                # Re, Im, Mag, Phase
            - image_defs['dynamic scan number']          # repetition
            - image_defs['label type']                   # ASL tag/control
            - image_defs['diffusion b value number']     # diffusion b value
            - image_defs['gradient orientation number']  # diffusion directoin
            - image_defs['cardiac phase number']         # cardiac phase
            - image_defs['echo number']                  # echo
            - image_defs['slice number']                 # slice

        Data sorting is done in two stages:

            1. an initial sort using the keys described above
            2. a resort after generating two additional sort keys:

                * a key to assign unique volume numbers to any volumes that
                  didn't have a unique sort based on the keys above
                  (see :func:`vol_numbers`).
                * a sort key based on `vol_is_full` to identify truncated
                  volumes

        A case where the initial sort may not create a unique label for each
        volume is diffusion scans acquired in the older V4 .PAR format, where
        diffusion direction info is not available.
        """
        idefs = self.image_defs
        slice_nos = idefs['slice number']
        dynamics = idefs['dynamic scan number']
        phases = idefs['cardiac phase number']
        echos = idefs['echo number']
        image_type = idefs['image_type_mr']
        asl_keys = (idefs['label type'],) if 'label type' in idefs.dtype.names else ()
        if self.general_info['diffusion'] != 0:
            bvals = self.get_def('diffusion b value number')
            if bvals is None:
                bvals = self.get_def('diffusion_b_factor')
            bvecs = self.get_def('gradient orientation number')
            if bvecs is None:
                diffusion_keys = (bvals,)
            else:
                diffusion_keys = (bvecs, bvals)
        else:
            diffusion_keys = ()
        keys = (slice_nos, echos, phases) + diffusion_keys + asl_keys + (dynamics, image_type)
        initial_sort_order = np.lexsort(keys)
        vol_nos = vol_numbers(slice_nos[initial_sort_order])
        is_full = vol_is_full(slice_nos[initial_sort_order], self.general_info['max_slices'])
        return initial_sort_order[np.lexsort((vol_nos, is_full))]

    def _lax_sort_order(self):
        """
        Sorts by (fast to slow): slice number, volume number.

        We calculate volume number by looking for repeating slice numbers (see
        :func:`vol_numbers`).
        """
        slice_nos = self.image_defs['slice number']
        is_full = vol_is_full(slice_nos, self.general_info['max_slices'])
        keys = (slice_nos, vol_numbers(slice_nos), np.logical_not(is_full))
        return np.lexsort(keys)

    def get_sorted_slice_indices(self):
        """Return indices to sort (and maybe discard) slices in REC file.

        If the recording is truncated, the returned indices take care of
        discarding any slice indices from incomplete volumes.

        If `self.strict_sort` is True, a more complicated sorting based on
        multiple fields from the .PAR file is used.  This may produce a
        different sort order than `strict_sort=False`, where volumes are sorted
        by the order in which the slices appear in the .PAR file.

        Returns
        -------
        slice_indices : list
            List for indexing into the last (third) dimension of the REC data
            array, and (equivalently) the only dimension of
            ``self.image_defs``.
        """
        if not self.strict_sort:
            sort_order = self._lax_sort_order()
        else:
            sort_order = self._strict_sort_order()
        n_used = np.prod(self.get_data_shape()[2:])
        return sort_order[:n_used]

    def get_volume_labels(self):
        """Dynamic labels corresponding to the final data dimension(s).

        This is useful for custom data sorting.  A subset of the info in
        ``self.image_defs`` is returned in an order that matches the final
        data dimension(s).  Only labels that have more than one unique value
        across the dataset will be returned.

        Returns
        -------
        sort_info : dict
            Each key corresponds to volume labels for a dynamically varying
            sequence dimension.  The ordering of the labels matches the volume
            ordering determined via ``self.get_sorted_slice_indices``.
        """
        sorted_indices = self.get_sorted_slice_indices()
        image_defs = self.image_defs
        dynamic_keys = ['cardiac phase number', 'echo number', 'label type', 'image_type_mr', 'dynamic scan number', 'scanning sequence', 'gradient orientation number', 'diffusion b value number']
        dynamic_keys = [d for d in dynamic_keys if d in image_defs.dtype.fields]
        non_unique_keys = []
        for key in dynamic_keys:
            ndim = image_defs[key].ndim
            if ndim == 1:
                num_unique = len(np.unique(image_defs[key]))
            else:
                raise ValueError('unexpected image_defs shape > 1D')
            if num_unique > 1:
                non_unique_keys.append(key)
        sl1_indices = image_defs['slice number'][sorted_indices] == 1
        sort_info = OrderedDict()
        for key in non_unique_keys:
            sort_info[key] = image_defs[key][sorted_indices][sl1_indices]
        return sort_info