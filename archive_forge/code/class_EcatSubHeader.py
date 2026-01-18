import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
class EcatSubHeader:
    _subhdrdtype = subhdr_dtype
    _data_type_codes = data_type_codes

    def __init__(self, hdr, mlist, fileobj):
        """parses the subheaders in the ecat (.v) file
        there is one subheader for each frame in the ecat file

        Parameters
        ----------
        hdr : EcatHeader
            ECAT main header
        mlist : array shape (N, 4)
            Matrix list
        fileobj : ECAT file <filename>.v  fileholder or file object
                  with read, seek methods
        """
        self._header = hdr
        self.endianness = hdr.endianness
        self._mlist = mlist
        self.fileobj = fileobj
        self.subheaders = read_subheaders(fileobj, mlist, hdr.endianness)

    def get_shape(self, frame=0):
        """returns shape of given frame"""
        subhdr = self.subheaders[frame]
        x = subhdr['x_dimension'].item()
        y = subhdr['y_dimension'].item()
        z = subhdr['z_dimension'].item()
        return (x, y, z)

    def get_nframes(self):
        """returns number of frames"""
        framed = get_frame_order(self._mlist)
        return len(framed)

    def _check_affines(self):
        """checks if all affines are equal across frames"""
        nframes = self.get_nframes()
        if nframes == 1:
            return True
        affs = [self.get_frame_affine(i) for i in range(nframes)]
        if affs:
            i = iter(affs)
            first = next(i)
            for item in i:
                if not np.allclose(first, item):
                    return False
        return True

    def get_frame_affine(self, frame=0):
        """returns best affine for given frame of data"""
        subhdr = self.subheaders[frame]
        x_off = subhdr['x_offset']
        y_off = subhdr['y_offset']
        z_off = subhdr['z_offset']
        zooms = self.get_zooms(frame=frame)
        dims = self.get_shape(frame)
        origin_offset = (np.array(dims) - 1) / 2.0
        aff = np.diag(zooms)
        aff[:3, -1] = -origin_offset * zooms[:-1] + np.array([x_off, y_off, z_off])
        return aff

    def get_zooms(self, frame=0):
        """returns zooms  ...pixdims"""
        subhdr = self.subheaders[frame]
        x_zoom = subhdr['x_pixel_size'] * 10
        y_zoom = subhdr['y_pixel_size'] * 10
        z_zoom = subhdr['z_pixel_size'] * 10
        return (x_zoom, y_zoom, z_zoom, 1)

    def _get_data_dtype(self, frame):
        dtcode = self.subheaders[frame]['data_type'].item()
        return self._data_type_codes.dtype[dtcode]

    def _get_frame_offset(self, frame=0):
        return int(self._mlist[frame][1] * BLOCK_SIZE)

    def _get_oriented_data(self, raw_data, orientation=None):
        """
        Get data oriented following ``patient_orientation`` header field. If
        the ``orientation`` parameter is given, return data according to this
        orientation.

        :param raw_data: Numpy array containing the raw data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing the oriented data
        """
        if orientation is None:
            orientation = self._header['patient_orientation']
        elif orientation == 'neurological':
            orientation = patient_orient_neurological[0]
        elif orientation == 'radiological':
            orientation = patient_orient_radiological[0]
        else:
            raise ValueError('orientation should be None, neurological or radiological')
        if orientation in patient_orient_neurological:
            raw_data = raw_data[::-1, ::-1, ::-1]
        elif orientation in patient_orient_radiological:
            raw_data = raw_data[:, ::-1, ::-1]
        return raw_data

    def raw_data_from_fileobj(self, frame=0, orientation=None):
        """
        Get raw data from file object.

        :param frame: Time frame index from where to fetch data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing (possibly oriented) raw data

        .. seealso:: data_from_fileobj
        """
        dtype = self._get_data_dtype(frame)
        if self._header.endianness is not native_code:
            dtype = dtype.newbyteorder(self._header.endianness)
        shape = self.get_shape(frame)
        offset = self._get_frame_offset(frame)
        fid_obj = self.fileobj
        raw_data = array_from_file(shape, dtype, fid_obj, offset=offset)
        raw_data = self._get_oriented_data(raw_data, orientation)
        return raw_data

    def data_from_fileobj(self, frame=0, orientation=None):
        """
        Read scaled data from file for a given frame

        :param frame: Time frame index from where to fetch data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing (possibly oriented) raw data

        .. seealso:: raw_data_from_fileobj
        """
        header = self._header
        subhdr = self.subheaders[frame]
        raw_data = self.raw_data_from_fileobj(frame, orientation)
        data = raw_data * header['ecat_calibration_factor'].item()
        data = data * subhdr['scale_factor'].item()
        return data