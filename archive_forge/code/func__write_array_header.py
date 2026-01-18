import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def _write_array_header(fp, d, version=None):
    """ Write the header for an array and returns the version used

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    version : tuple or None
        None means use oldest that works. Providing an explicit version will
        raise a ValueError if the format does not allow saving this data.
        Default: None
    """
    header = ['{']
    for key, value in sorted(d.items()):
        header.append("'%s': %s, " % (key, repr(value)))
    header.append('}')
    header = ''.join(header)
    shape = d['shape']
    header += ' ' * (GROWTH_AXIS_MAX_DIGITS - len(repr(shape[-1 if d['fortran_order'] else 0])) if len(shape) > 0 else 0)
    if version is None:
        header = _wrap_header_guess_version(header)
    else:
        header = _wrap_header(header, version)
    fp.write(header)