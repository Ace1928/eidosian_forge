import sys
import os
import struct
import logging
import numpy as np
class SimpleDicomReader(object):
    """
    This class provides reading of pixel data from DICOM files. It is
    focussed on getting the pixel data, not the meta info.

    To use, first create an instance of this class (giving it
    a file object or filename). Next use the info attribute to
    get a dict of the meta data. The loading of pixel data is
    deferred until get_numpy_array() is called.

    Comparison with Pydicom
    -----------------------

    This code focusses on getting the pixel data out, which allows some
    shortcuts, resulting in the code being much smaller.

    Since the processing of data elements is much cheaper (it skips a lot
    of tags), this code is about 3x faster than pydicom (except for the
    deflated DICOM files).

    This class does borrow some code (and ideas) from the pydicom
    project, and (to the best of our knowledge) has the same limitations
    as pydicom with regard to the type of files that it can handle.

    Limitations
    -----------

    For more advanced DICOM processing, please check out pydicom.

      * Only a predefined subset of data elements (meta information) is read.
      * This is a reader; it can not write DICOM files.
      * (just like pydicom) it can handle none of the compressed DICOM
        formats except for "Deflated Explicit VR Little Endian"
        (1.2.840.10008.1.2.1.99).

    """

    def __init__(self, file):
        if isinstance(file, str):
            self._filename = file
            self._file = open(file, 'rb')
        else:
            self._filename = '<unknown file>'
            self._file = file
        self._pixel_data_loc = None
        self.is_implicit_VR = False
        self.is_little_endian = True
        self._unpackPrefix = '<'
        self._info = {}
        self._converters = {'US': lambda x: self._unpack('H', x), 'UL': lambda x: self._unpack('L', x), 'DS': lambda x: self._splitValues(x, float, '\\'), 'IS': lambda x: self._splitValues(x, int, '\\'), 'AS': lambda x: x.decode('ascii', 'ignore').strip('\x00'), 'DA': lambda x: x.decode('ascii', 'ignore').strip('\x00'), 'TM': lambda x: x.decode('ascii', 'ignore').strip('\x00'), 'UI': lambda x: x.decode('ascii', 'ignore').strip('\x00'), 'LO': lambda x: x.decode('utf-8', 'ignore').strip('\x00').rstrip(), 'CS': lambda x: self._splitValues(x, float, '\\'), 'PN': lambda x: x.decode('utf-8', 'ignore').strip('\x00').rstrip()}
        self._read()

    @property
    def info(self):
        return self._info

    def _splitValues(self, x, type, splitter):
        s = x.decode('ascii').strip('\x00')
        try:
            if splitter in s:
                return tuple([type(v) for v in s.split(splitter) if v.strip()])
            else:
                return type(s)
        except ValueError:
            return s

    def _unpack(self, fmt, value):
        return struct.unpack(self._unpackPrefix + fmt, value)[0]

    def __iter__(self):
        return iter(self._info.keys())

    def __getattr__(self, key):
        info = object.__getattribute__(self, '_info')
        if key in info:
            return info[key]
        return object.__getattribute__(self, key)

    def _read(self):
        f = self._file
        f.seek(128)
        if f.read(4) != b'DICM':
            raise NotADicomFile('Not a valid DICOM file.')
        self._read_header()
        self._read_data_elements()
        self._get_shape_and_sampling()
        if os.path.isfile(self._filename):
            self._file.close()
            self._file = None

    def _readDataElement(self):
        f = self._file
        group = self._unpack('H', f.read(2))
        element = self._unpack('H', f.read(2))
        if self.is_implicit_VR:
            vl = self._unpack('I', f.read(4))
        else:
            vr = f.read(2)
            if vr in (b'OB', b'OW', b'SQ', b'UN'):
                reserved = f.read(2)
                vl = self._unpack('I', f.read(4))
            else:
                vl = self._unpack('H', f.read(2))
        if group == 32736 and element == 16:
            here = f.tell()
            self._pixel_data_loc = (here, vl)
            f.seek(here + vl)
            return (group, element, b'Deferred loading of pixel data')
        else:
            if vl == 4294967295:
                value = self._read_undefined_length_value()
            else:
                value = f.read(vl)
            return (group, element, value)

    def _read_undefined_length_value(self, read_size=128):
        """Copied (in compacted form) from PyDicom
        Copyright Darcy Mason.
        """
        fp = self._file
        search_rewind = 3
        bytes_to_find = struct.pack(self._unpackPrefix + 'HH', SequenceDelimiterTag[0], SequenceDelimiterTag[1])
        found = False
        value_chunks = []
        while not found:
            chunk_start = fp.tell()
            bytes_read = fp.read(read_size)
            if len(bytes_read) < read_size:
                new_bytes = fp.read(read_size - len(bytes_read))
                bytes_read += new_bytes
                if len(bytes_read) < read_size:
                    raise EOFError('End of file reached before sequence delimiter found.')
            index = bytes_read.find(bytes_to_find)
            if index != -1:
                found = True
                value_chunks.append(bytes_read[:index])
                fp.seek(chunk_start + index + 4)
                length = fp.read(4)
                if length != b'\x00\x00\x00\x00':
                    logger.warning('Expected 4 zero bytes after undefined length delimiter')
            else:
                fp.seek(fp.tell() - search_rewind)
                value_chunks.append(bytes_read[:-search_rewind])
        return b''.join(value_chunks)

    def _read_header(self):
        f = self._file
        TransferSyntaxUID = None
        try:
            while True:
                fp_save = f.tell()
                group, element, value = self._readDataElement()
                if group == 2:
                    if group == 2 and element == 16:
                        TransferSyntaxUID = value.decode('ascii').strip('\x00')
                else:
                    f.seek(fp_save)
                    break
        except (EOFError, struct.error):
            raise RuntimeError('End of file reached while still in header.')
        self._info['TransferSyntaxUID'] = TransferSyntaxUID
        if TransferSyntaxUID is None:
            is_implicit_VR, is_little_endian = (False, True)
        elif TransferSyntaxUID == '1.2.840.10008.1.2.1':
            is_implicit_VR, is_little_endian = (False, True)
        elif TransferSyntaxUID == '1.2.840.10008.1.2.2':
            is_implicit_VR, is_little_endian = (False, False)
        elif TransferSyntaxUID == '1.2.840.10008.1.2':
            is_implicit_VR, is_little_endian = (True, True)
        elif TransferSyntaxUID == '1.2.840.10008.1.2.1.99':
            is_implicit_VR, is_little_endian = (False, True)
            self._inflate()
        else:
            t, extra_info = (TransferSyntaxUID, '')
            if '1.2.840.10008.1.2.4.50' <= t < '1.2.840.10008.1.2.4.99':
                extra_info = ' (JPEG)'
            if '1.2.840.10008.1.2.4.90' <= t < '1.2.840.10008.1.2.4.99':
                extra_info = ' (JPEG 2000)'
            if t == '1.2.840.10008.1.2.5':
                extra_info = ' (RLE)'
            if t == '1.2.840.10008.1.2.6.1':
                extra_info = ' (RFC 2557)'
            raise CompressedDicom('The dicom reader can only read files with uncompressed image data - not %r%s. You can try using dcmtk or gdcm to convert the image.' % (t, extra_info))
        self.is_implicit_VR = is_implicit_VR
        self.is_little_endian = is_little_endian
        self._unpackPrefix = '><'[is_little_endian]

    def _read_data_elements(self):
        info = self._info
        try:
            while True:
                group, element, value = self._readDataElement()
                if group in GROUPS:
                    key = (group, element)
                    name, vr = MINIDICT.get(key, (None, None))
                    if name:
                        converter = self._converters.get(vr, lambda x: x)
                        info[name] = converter(value)
        except (EOFError, struct.error):
            pass

    def get_numpy_array(self):
        """Get numpy arra for this DICOM file, with the correct shape,
        and pixel values scaled appropriately.
        """
        if 'PixelData' not in self:
            raise TypeError('No pixel data found in this dataset.')
        if self._pixel_data_loc and len(self.PixelData) < 100:
            close_file = False
            if self._file is None:
                close_file = True
                self._file = open(self._filename, 'rb')
            self._file.seek(self._pixel_data_loc[0])
            if self._pixel_data_loc[1] == 4294967295:
                value = self._read_undefined_length_value()
            else:
                value = self._file.read(self._pixel_data_loc[1])
            if close_file:
                self._file.close()
                self._file = None
            self._info['PixelData'] = value
        data = self._pixel_data_numpy()
        data = self._apply_slope_and_offset(data)
        self._info['PixelData'] = b'Data converted to numpy array, ' + b'raw data removed to preserve memory'
        return data

    def _get_shape_and_sampling(self):
        """Get shape and sampling without actuall using the pixel data.
        In this way, the user can get an idea what's inside without having
        to load it.
        """
        if 'NumberOfFrames' in self and self.NumberOfFrames > 1:
            if self.SamplesPerPixel > 1:
                shape = (self.SamplesPerPixel, self.NumberOfFrames, self.Rows, self.Columns)
            else:
                shape = (self.NumberOfFrames, self.Rows, self.Columns)
        elif 'SamplesPerPixel' in self:
            if self.SamplesPerPixel > 1:
                if self.BitsAllocated == 8:
                    shape = (self.SamplesPerPixel, self.Rows, self.Columns)
                else:
                    raise NotImplementedError('DICOM plugin only handles SamplesPerPixel > 1 if Bits Allocated = 8')
            else:
                shape = (self.Rows, self.Columns)
        else:
            raise RuntimeError('DICOM file has no SamplesPerPixel (perhaps this is a report?)')
        if 'PixelSpacing' in self:
            sampling = (float(self.PixelSpacing[0]), float(self.PixelSpacing[1]))
        else:
            sampling = (1.0, 1.0)
        if 'SliceSpacing' in self:
            sampling = (abs(self.SliceSpacing),) + sampling
        sampling = (1.0,) * (len(shape) - len(sampling)) + sampling[-len(shape):]
        self._info['shape'] = shape
        self._info['sampling'] = sampling

    def _pixel_data_numpy(self):
        """Return a NumPy array of the pixel data."""
        if 'PixelData' not in self:
            raise TypeError('No pixel data found in this dataset.')
        need_byteswap = self.is_little_endian != sys_is_little_endian
        format_str = '%sint%d' % (('u', '')[self.PixelRepresentation], self.BitsAllocated)
        try:
            numpy_format = np.dtype(format_str)
        except TypeError:
            raise TypeError("Data type not understood by NumPy: format='%s',  PixelRepresentation=%d, BitsAllocated=%d" % (numpy_format, self.PixelRepresentation, self.BitsAllocated))
        arr = np.frombuffer(self.PixelData, numpy_format).copy()
        if need_byteswap:
            arr.byteswap(True)
        arr = arr.reshape(*self._info['shape'])
        return arr

    def _apply_slope_and_offset(self, data):
        """
        If RescaleSlope and RescaleIntercept are present in the data,
        apply them. The data type of the data is changed if necessary.
        """
        slope, offset = (1, 0)
        needFloats, needApplySlopeOffset = (False, False)
        if 'RescaleSlope' in self:
            needApplySlopeOffset = True
            slope = self.RescaleSlope
        if 'RescaleIntercept' in self:
            needApplySlopeOffset = True
            offset = self.RescaleIntercept
        if int(slope) != slope or int(offset) != offset:
            needFloats = True
        if not needFloats:
            slope, offset = (int(slope), int(offset))
        if needApplySlopeOffset:
            if data.dtype in [np.float32, np.float64]:
                pass
            elif needFloats:
                data = data.astype(np.float32)
            else:
                minReq, maxReq = (data.min(), data.max())
                minReq = min([minReq, minReq * slope + offset, maxReq * slope + offset])
                maxReq = max([maxReq, minReq * slope + offset, maxReq * slope + offset])
                dtype = None
                if minReq < 0:
                    maxReq = max([-minReq, maxReq])
                    if maxReq < 2 ** 7:
                        dtype = np.int8
                    elif maxReq < 2 ** 15:
                        dtype = np.int16
                    elif maxReq < 2 ** 31:
                        dtype = np.int32
                    else:
                        dtype = np.float32
                elif maxReq < 2 ** 8:
                    dtype = np.int8
                elif maxReq < 2 ** 16:
                    dtype = np.int16
                elif maxReq < 2 ** 32:
                    dtype = np.int32
                else:
                    dtype = np.float32
                if dtype != data.dtype:
                    data = data.astype(dtype)
            data *= slope
            data += offset
        return data

    def _inflate(self):
        import zlib
        from io import BytesIO
        zipped = self._file.read()
        unzipped = zlib.decompress(zipped, -zlib.MAX_WBITS)
        self._file = BytesIO(unzipped)