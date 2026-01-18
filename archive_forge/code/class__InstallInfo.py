import inspect
import io
import os
import platform
import warnings
import numpy
import cupy
import cupy_backends
class _InstallInfo(object):

    def __init__(self):
        cupy_package_root = self._get_cupy_package_root()
        if cupy_package_root is not None:
            data_root = os.path.join(cupy_package_root, '.data')
            data_paths = {'lib': _dir_or_none(os.path.join(data_root, 'lib')), 'include': _dir_or_none(os.path.join(data_root, 'include'))}
        else:
            data_paths = {'lib': None, 'include': None}
        self.cupy_package_root = cupy_package_root
        self.data_paths = data_paths

    def get_data_path(self, data_type):
        if data_type not in self.data_paths:
            raise ValueError('Invalid data type: {}'.format(data_type))
        return self.data_paths[data_type]

    def _get_cupy_package_root(self):
        try:
            cupy_path = inspect.getfile(cupy)
        except TypeError:
            return None
        return os.path.dirname(cupy_path)