import ctypes
import itertools
import numpy as np
from . import QT_LIB, QtCore, QtGui, compat
class PrimitiveArray:

    def __init__(self, Klass, nfields, *, use_array=None):
        self._Klass = Klass
        self._nfields = nfields
        self._capa = -1
        self.use_sip_array = False
        self.use_ptr_to_array = False
        if QT_LIB.startswith('PyQt'):
            if use_array is None:
                use_array = hasattr(sip, 'array') and (393985 <= QtCore.PYQT_VERSION or 331527 <= QtCore.PYQT_VERSION < 393216)
            self.use_sip_array = use_array
        if QT_LIB.startswith('PySide'):
            if use_array is None:
                use_array = Klass is QtGui.QPainter.PixmapFragment or pyside_version_info >= (6, 4, 3)
            self.use_ptr_to_array = use_array
        self.resize(0)

    def resize(self, size):
        if self.use_sip_array:
            if sip.SIP_VERSION >= 395016:
                if size <= self._capa:
                    self._size = size
                    return
            elif size == self._capa:
                return
            self._siparray = sip.array(self._Klass, size)
        else:
            if size <= self._capa:
                self._size = size
                return
            self._ndarray = np.empty((size, self._nfields), dtype=np.float64)
            if self.use_ptr_to_array:
                self._objs = None
            else:
                self._objs = self._wrap_instances(self._ndarray)
        self._capa = size
        self._size = size

    def _wrap_instances(self, array):
        return list(map(compat.wrapinstance, itertools.count(array.ctypes.data, array.strides[0]), itertools.repeat(self._Klass, array.shape[0])))

    def __len__(self):
        return self._size

    def ndarray(self):
        if self.use_sip_array:
            if sip.SIP_VERSION >= 395016 and np.__version__ != '1.22.4':
                mv = self._siparray
            else:
                mv = sip.voidptr(self._siparray, self._capa * self._nfields * 8)
            nd = np.frombuffer(mv, dtype=np.float64, count=self._size * self._nfields)
            return nd.reshape((-1, self._nfields))
        else:
            return self._ndarray[:self._size]

    def instances(self):
        if self.use_sip_array:
            if self._size == self._capa:
                return self._siparray
            else:
                return self._siparray[:self._size]
        if self._objs is None:
            self._objs = self._wrap_instances(self._ndarray)
        if self._size == self._capa:
            return self._objs
        else:
            return self._objs[:self._size]

    def drawargs(self):
        if self.use_ptr_to_array:
            if self._capa > 0:
                ptr = compat.wrapinstance(self._ndarray.ctypes.data, self._Klass)
            else:
                ptr = None
            return (ptr, self._size)
        else:
            return (self.instances(),)