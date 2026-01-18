import logging
from rasterio._base import get_dataset_driver, driver_can_create, driver_can_create_copy
from rasterio._io import (
from rasterio.windows import WindowMethodsMixin
from rasterio.env import Env, ensure_env
from rasterio.transform import TransformMethodsMixin
from rasterio._path import _UnparsedPath
class ZipMemoryFile(MemoryFile):
    """A read-only BytesIO-like object backed by an in-memory zip file.

    This allows a zip file containing formatted files to be read
    without I/O.
    """

    def __init__(self, file_or_bytes=None):
        super().__init__(file_or_bytes, ext='zip')

    @ensure_env
    def open(self, path, driver=None, sharing=False, **kwargs):
        """Open a dataset within the zipped stream.

        Parameters
        ----------
        path : str
            Path to a dataset in the zip file, relative to the root of the
            archive.

        Other parameters are optional and have the same semantics as the
        parameters of `rasterio.open()`.

        Returns
        -------
        A Rasterio dataset object
        """
        zippath = _UnparsedPath('/vsizip{0}/{1}'.format(self.name, path.lstrip('/')))
        if self.closed:
            raise OSError('I/O operation on closed file.')
        return DatasetReader(zippath, driver=driver, sharing=sharing, **kwargs)