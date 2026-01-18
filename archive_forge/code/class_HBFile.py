import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
class HBFile:

    def __init__(self, file, hb_info=None):
        """Create a HBFile instance.

        Parameters
        ----------
        file : file-object
            StringIO work as well
        hb_info : HBInfo, optional
            Should be given as an argument for writing, in which case the file
            should be writable.
        """
        self._fid = file
        if hb_info is None:
            self._hb_info = HBInfo.from_file(file)
        else:
            self._hb_info = hb_info

    @property
    def title(self):
        return self._hb_info.title

    @property
    def key(self):
        return self._hb_info.key

    @property
    def type(self):
        return self._hb_info.mxtype.value_type

    @property
    def structure(self):
        return self._hb_info.mxtype.structure

    @property
    def storage(self):
        return self._hb_info.mxtype.storage

    def read_matrix(self):
        return _read_hb_data(self._fid, self._hb_info)

    def write_matrix(self, m):
        return _write_data(m, self._fid, self._hb_info)