import copy
import os
import pickle
import warnings
import numpy as np
def _readHDF5(self, fileName, readAllData=None, writable=False, **kargs):
    if 'close' in kargs and readAllData is None:
        readAllData = kargs['close']
    if readAllData is True and writable is True:
        raise Exception('Incompatible arguments: readAllData=True and writable=True')
    if not HAVE_HDF5:
        try:
            assert writable == False
            assert readAllData != False
            self._readHDF5Remote(fileName)
            return
        except:
            raise Exception("The file '%s' is HDF5-formatted, but the HDF5 library (h5py) was not found." % fileName)
    if readAllData is None:
        size = os.stat(fileName).st_size
        readAllData = size < 500000000.0
    if writable is True:
        mode = 'r+'
    else:
        mode = 'r'
    f = h5py.File(fileName, mode)
    ver = f.attrs['MetaArray']
    try:
        ver = ver.decode('utf-8')
    except:
        pass
    if ver > MetaArray.version:
        print('Warning: This file was written with MetaArray version %s, but you are using version %s. (Will attempt to read anyway)' % (str(ver), str(MetaArray.version)))
    meta = MetaArray.readHDF5Meta(f['info'])
    self._info = meta
    if writable or not readAllData:
        self._data = f['data']
        self._openFile = f
    else:
        self._data = f['data'][:]
        f.close()