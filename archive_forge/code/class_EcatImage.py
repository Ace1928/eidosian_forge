import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
class EcatImage(SpatialImage):
    """Class returns a list of Ecat images, with one image(hdr/data) per frame"""
    header_class = EcatHeader
    subheader_class = EcatSubHeader
    valid_exts = ('.v',)
    files_types = (('image', '.v'), ('header', '.v'))
    header: EcatHeader
    _subheader: EcatSubHeader
    ImageArrayProxy = EcatImageArrayProxy

    def __init__(self, dataobj, affine, header, subheader, mlist, extra=None, file_map=None):
        """Initialize Image

        The image is a combination of
        (array, affine matrix, header, subheader, mlist)
        with optional meta data in `extra`, and filename / file-like objects
        contained in the `file_map`.

        Parameters
        ----------
        dataobj : array-like
            image data
        affine : None or (4,4) array-like
            homogeneous affine giving relationship between voxel coords and
            world coords.
        header : None or header instance
            meta data for this image format
        subheader : None or subheader instance
            meta data for each sub-image for frame in the image
        mlist : None or array
            Matrix list array giving offset and order of data in file
        extra : None or mapping, optional
            metadata associated with this image that cannot be
            stored in header or subheader
        file_map : mapping, optional
            mapping giving file information for this image format

        Examples
        --------
        >>> import os
        >>> import nibabel as nib
        >>> nibabel_dir = os.path.dirname(nib.__file__)
        >>> from nibabel import ecat
        >>> ecat_file = os.path.join(nibabel_dir,'tests','data','tinypet.v')
        >>> img = ecat.load(ecat_file)
        >>> frame0 = img.get_frame(0)
        >>> frame0.shape == (10, 10, 3)
        True
        >>> data4d = img.get_fdata()
        >>> data4d.shape == (10, 10, 3, 1)
        True
        """
        self._subheader = subheader
        self._mlist = mlist
        self._dataobj = dataobj
        if affine is not None:
            affine = np.array(affine, dtype=np.float64, copy=True)
            if not affine.shape == (4, 4):
                raise ValueError('Affine should be shape 4,4')
        self._affine = affine
        if extra is None:
            extra = {}
        self.extra = extra
        self._header = header
        if file_map is None:
            file_map = self.__class__.make_file_map()
        self.file_map = file_map
        self._data_cache = None
        self._fdata_cache = None

    @property
    def affine(self):
        if not self._subheader._check_affines():
            warnings.warn('Affines different across frames, loading affine from FIRST frame', UserWarning)
        return self._affine

    def get_frame_affine(self, frame):
        """returns 4X4 affine"""
        return self._subheader.get_frame_affine(frame=frame)

    def get_frame(self, frame, orientation=None):
        """
        Get full volume for a time frame

        :param frame: Time frame index from where to fetch data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing (possibly oriented) raw data
        """
        return self._subheader.data_from_fileobj(frame, orientation)

    def get_data_dtype(self, frame):
        subhdr = self._subheader
        dt = subhdr._get_data_dtype(frame)
        return dt

    @property
    def shape(self):
        x, y, z = self._subheader.get_shape()
        nframes = self._subheader.get_nframes()
        return (x, y, z, nframes)

    def get_mlist(self):
        """get access to the mlist"""
        return self._mlist

    def get_subheaders(self):
        """get access to subheaders"""
        return self._subheader

    @staticmethod
    def _get_fileholders(file_map):
        """returns files specific to header and image of the image
        for ecat .v this is the same image file

        Returns
        -------
        header : file holding header data
        image : file holding image data
        """
        return (file_map['header'], file_map['image'])

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """class method to create image from mapping
        specified in file_map
        """
        hdr_file, img_file = klass._get_fileholders(file_map)
        hdr_fid = hdr_file.get_prepare_fileobj(mode='rb')
        header = klass.header_class.from_fileobj(hdr_fid)
        hdr_copy = header.copy()
        mlist = np.zeros((header['num_frames'], 4), dtype=np.int32)
        mlist_data = read_mlist(hdr_fid, hdr_copy.endianness)
        mlist[:len(mlist_data)] = mlist_data
        subheaders = klass.subheader_class(hdr_copy, mlist, hdr_fid)
        data = klass.ImageArrayProxy(subheaders)
        if not subheaders._check_affines():
            warnings.warn('Affines different across frames, loading affine from FIRST frame', UserWarning)
        aff = subheaders.get_frame_affine()
        img = klass(data, aff, header, subheaders, mlist, extra=None, file_map=file_map)
        return img

    def _get_empty_dir(self):
        """
        Get empty directory entry of the form
        [numAvail, nextDir, previousDir, numUsed]
        """
        return np.array([31, 2, 0, 0], dtype=np.int32)

    def _write_data(self, data, stream, pos, dtype=None, endianness=None):
        """
        Write data to ``stream`` using an array_writer

        :param data: Numpy array containing the dat
        :param stream: The file-like object to write the data to
        :param pos: The position in the stream to write the data to
        :param endianness: Endianness code of the data to write
        """
        if dtype is None:
            dtype = data.dtype
        if endianness is None:
            endianness = native_code
        stream.seek(pos)
        make_array_writer(data.view(data.dtype.newbyteorder(endianness)), dtype).to_fileobj(stream)

    def to_file_map(self, file_map=None):
        """Write ECAT7 image to `file_map` or contained ``self.file_map``

        The format consist of:

        - A main header (512L) with dictionary entries in the form
            [numAvail, nextDir, previousDir, numUsed]
        - For every frame (3D volume in 4D data)
          - A subheader (size = frame_offset)
          - Frame data (3D volume)
        """
        if file_map is None:
            file_map = self.file_map
        self.get_fdata()
        hdr = self.header
        mlist = self._mlist
        subheaders = self.get_subheaders()
        dir_pos = 512
        entry_pos = dir_pos + 16
        current_dir = self._get_empty_dir()
        hdr_fh, img_fh = self._get_fileholders(file_map)
        hdrf = hdr_fh.get_prepare_fileobj(mode='wb')
        imgf = hdrf
        hdr.write_to(hdrf)
        for index in range(0, self.header['num_frames']):
            frame_offset = subheaders._get_frame_offset(index) - 512
            imgf.seek(frame_offset)
            subhdr = subheaders.subheaders[index]
            imgf.write(subhdr.tobytes())
            pos = imgf.tell()
            imgf.seek(pos + 2)
            image = self._subheader.raw_data_from_fileobj(index)
            self._write_data(image, imgf, pos + 2, endianness='>')
            self._write_data(mlist[index], imgf, entry_pos, endianness='>')
            entry_pos = entry_pos + 16
            current_dir[0] = current_dir[0] - 1
            current_dir[3] = current_dir[3] + 1
            if current_dir[0] == 0:
                self._write_data(current_dir, imgf, dir_pos)
                current_dir = self._get_empty_dir()
                current_dir[3] = dir_pos / 512
                dir_pos = mlist[index][2] + 1
                entry_pos = dir_pos + 16
        tmp_avail = current_dir[0]
        tmp_used = current_dir[3]
        while current_dir[0] > 0:
            entry_pos = dir_pos + 16 + 16 * current_dir[3]
            self._write_data(np.zeros(4, dtype=np.int32), imgf, entry_pos)
            current_dir[0] = current_dir[0] - 1
            current_dir[3] = current_dir[3] + 1
        current_dir[0] = tmp_avail
        current_dir[3] = tmp_used
        self._write_data(current_dir, imgf, dir_pos, endianness='>')

    @classmethod
    def from_image(klass, img):
        raise NotImplementedError('Ecat images can only be generated from file objects')

    @classmethod
    def load(klass, filespec):
        return klass.from_filename(filespec)