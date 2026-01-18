import os
import sys
import logging
import subprocess
from ..core import Format, BaseProgressIndicator, StdoutProgressIndicator
from ..core import read_n_bytes
class DicomFormat(Format):
    """See :mod:`imageio.plugins.dicom`"""

    def _can_read(self, request):
        if os.path.isdir(request.filename):
            files = os.listdir(request.filename)
            for fname in sorted(files):
                filename = os.path.join(request.filename, fname)
                if os.path.isfile(filename) and 'DICOMDIR' not in fname:
                    with open(filename, 'rb') as f:
                        first_bytes = read_n_bytes(f, 140)
                    return first_bytes[128:132] == b'DICM'
            else:
                return False
        return request.firstbytes[128:132] == b'DICM'

    def _can_write(self, request):
        return False

    class Reader(Format.Reader):
        _compressed_warning_dirs = set()

        def _open(self, progress=True):
            if not _dicom:
                load_lib()
            if os.path.isdir(self.request.filename):
                self._info = {}
                self._data = None
            else:
                try:
                    dcm = _dicom.SimpleDicomReader(self.request.get_file())
                except _dicom.CompressedDicom as err:
                    cmd = get_gdcmconv_exe()
                    if not cmd and 'JPEG' in str(err):
                        cmd = get_dcmdjpeg_exe()
                    if not cmd:
                        msg = err.args[0].replace('using', 'installing')
                        msg = msg.replace('convert', 'auto-convert')
                        err.args = (msg,)
                        raise
                    else:
                        fname1 = self.request.get_local_filename()
                        fname2 = fname1 + '.raw'
                        try:
                            subprocess.check_call(cmd + [fname1, fname2])
                        except Exception:
                            raise err
                        d = os.path.dirname(fname1)
                        if d not in self._compressed_warning_dirs:
                            self._compressed_warning_dirs.add(d)
                            logger.warning('DICOM file contained compressed data. ' + 'Autoconverting with ' + cmd[0] + ' (this warning is shown once for each directory)')
                        dcm = _dicom.SimpleDicomReader(fname2)
                self._info = dcm._info
                self._data = dcm.get_numpy_array()
            self._series = None
            if isinstance(progress, BaseProgressIndicator):
                self._progressIndicator = progress
            elif progress is True:
                p = StdoutProgressIndicator('Reading DICOM')
                self._progressIndicator = p
            elif progress in (None, False):
                self._progressIndicator = BaseProgressIndicator('Dummy')
            else:
                raise ValueError('Invalid value for progress.')

        def _close(self):
            self._info = None
            self._data = None
            self._series = None

        @property
        def series(self):
            if self._series is None:
                pi = self._progressIndicator
                self._series = _dicom.process_directory(self.request, pi)
            return self._series

        def _get_length(self):
            if self._data is None:
                dcm = self.series[0][0]
                self._info = dcm._info
                self._data = dcm.get_numpy_array()
            nslices = self._data.shape[0] if self._data.ndim == 3 else 1
            if self.request.mode[1] == 'i':
                return nslices
            elif self.request.mode[1] == 'I':
                if nslices > 1:
                    return nslices
                else:
                    return sum([len(serie) for serie in self.series])
            elif self.request.mode[1] == 'v':
                if nslices > 1:
                    return 1
                else:
                    return len(self.series)
            elif self.request.mode[1] == 'V':
                return len(self.series)
            else:
                raise RuntimeError('DICOM plugin should know what to expect.')

        def _get_slice_data(self, index):
            nslices = self._data.shape[0] if self._data.ndim == 3 else 1
            if nslices > 1:
                return (self._data[index], self._info)
            elif index == 0:
                return (self._data, self._info)
            else:
                raise IndexError('Dicom file contains only one slice.')

        def _get_data(self, index):
            if self._data is None:
                dcm = self.series[0][0]
                self._info = dcm._info
                self._data = dcm.get_numpy_array()
            nslices = self._data.shape[0] if self._data.ndim == 3 else 1
            if self.request.mode[1] == 'i':
                return self._get_slice_data(index)
            elif self.request.mode[1] == 'I':
                if index == 0 and nslices > 1:
                    return (self._data[index], self._info)
                else:
                    L = []
                    for serie in self.series:
                        L.extend([dcm_ for dcm_ in serie])
                    return (L[index].get_numpy_array(), L[index].info)
            elif self.request.mode[1] in 'vV':
                if index == 0 and nslices > 1:
                    return (self._data, self._info)
                else:
                    return (self.series[index].get_numpy_array(), self.series[index].info)
            elif len(self.series) > 1:
                return (self.series[index].get_numpy_array(), self.series[index].info)
            else:
                return self._get_slice_data(index)

        def _get_meta_data(self, index):
            if self._data is None:
                dcm = self.series[0][0]
                self._info = dcm._info
                self._data = dcm.get_numpy_array()
            nslices = self._data.shape[0] if self._data.ndim == 3 else 1
            if index is None:
                return self._info
            if self.request.mode[1] == 'i':
                return self._info
            elif self.request.mode[1] == 'I':
                if index == 0 and nslices > 1:
                    return self._info
                else:
                    L = []
                    for serie in self.series:
                        L.extend([dcm_ for dcm_ in serie])
                    return L[index].info
            elif self.request.mode[1] in 'vV':
                if index == 0 and nslices > 1:
                    return self._info
                else:
                    return self.series[index].info
            else:
                raise ValueError('DICOM plugin should know what to expect.')