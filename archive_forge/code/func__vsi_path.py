import pathlib
import re
import sys
from urllib.parse import urlparse
import attr
from rasterio.errors import PathError
def _vsi_path(path):
    """Convert a parsed path to a GDAL VSI path

    Parameters
    ----------
    path : Path
        A ParsedPath or UnparsedPath object.

    Returns
    -------
    str

    """
    if isinstance(path, _UnparsedPath):
        return path.path
    elif isinstance(path, _ParsedPath):
        if not path.scheme:
            return path.path
        else:
            if path.scheme.split('+')[-1] in CURLSCHEMES:
                suffix = '{}://'.format(path.scheme.split('+')[-1])
            else:
                suffix = ''
            prefix = '/'.join(('vsi{0}'.format(SCHEMES[p]) for p in path.scheme.split('+') if p != 'file'))
            if prefix:
                if path.archive:
                    result = '/{}/{}{}/{}'.format(prefix, suffix, path.archive, path.path.lstrip('/'))
                else:
                    result = '/{}/{}{}'.format(prefix, suffix, path.path)
            else:
                result = path.path
            return result
    else:
        raise ValueError('path must be a ParsedPath or UnparsedPath object')