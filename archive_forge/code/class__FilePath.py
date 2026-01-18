import logging
from rasterio._base import get_dataset_driver, driver_can_create, driver_can_create_copy
from rasterio._io import (
from rasterio.windows import WindowMethodsMixin
from rasterio.env import Env, ensure_env
from rasterio.transform import TransformMethodsMixin
from rasterio._path import _UnparsedPath
class _FilePath(FilePathBase):
    """A BytesIO-like object, backed by a Python file object.

    Examples
    --------

    A GeoTIFF can be loaded in memory and accessed using the GeoTIFF
    format driver

    >>> with open('tests/data/RGB.byte.tif', 'rb') as f, FilePath(f) as vsi_file:
    ...     with vsi_file.open() as src:
    ...         pprint.pprint(src.profile)
    ...
    {'count': 3,
     'crs': CRS({'init': 'epsg:32618'}),
     'driver': 'GTiff',
     'dtype': 'uint8',
     'height': 718,
     'interleave': 'pixel',
     'nodata': 0.0,
     'tiled': False,
     'transform': Affine(300.0379266750948, 0.0, 101985.0,
           0.0, -300.041782729805, 2826915.0),
     'width': 791}

    """

    def __init__(self, filelike_obj, dirname=None, filename=None):
        """Create a new wrapper around the provided file-like object.

        Parameters
        ----------
        filelike_obj : file-like object
            Open file-like object. Currently only reading is supported.
        filename : str, optional
            An optional filename. A unique one will otherwise be generated.

        Returns
        -------
        PythonVSIFile
        """
        super().__init__(filelike_obj, dirname=dirname, filename=filename)

    @ensure_env
    def open(self, driver=None, sharing=False, **kwargs):
        """Open the file and return a Rasterio dataset object.

        The provided file-like object is assumed to be readable.
        Writing is currently not supported.

        Parameters are optional and have the same semantics as the
        parameters of `rasterio.open()`.

        Returns
        -------
        DatasetReader

        Raises
        ------
        IOError
            If the memory file is closed.

        """
        mempath = _UnparsedPath(self.name)
        if self.closed:
            raise IOError('I/O operation on closed file.')
        log.debug('VSI path: {}'.format(mempath.path))
        return DatasetReader(mempath, driver=driver, sharing=sharing, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()