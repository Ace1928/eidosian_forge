import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def _read_annot_ctab_new_format(fobj, ctab_version):
    """Read in a new-style Freesurfer color table from `fobj`.

    Note that the output color table ``ctab`` is in RGBT form, where T
    (transparency) is 255 - alpha.

    This function is used by :func:`read_annot`.

    Parameters
    ----------

    fobj : file-like
        Open file handle to a Freesurfer `.annot` file, with seek point
        at the beginning of the color table data.
    ctab_version : int
        Color table format version - must be equal to 2

    Returns
    -------

    ctab : ndarray, shape (n_labels, 5)
        RGBT colortable array - the last column contains all zeros.
    names : list of str
        The names of the labels. The length of the list is n_labels.
    """
    assert hasattr(fobj, 'read')
    dt = _ANNOT_DT
    if ctab_version != 2:
        raise Exception('Unrecognised .annot file version (%i)', ctab_version)
    max_index = np.fromfile(fobj, dt, 1)[0]
    ctab = np.zeros((max_index, 5), dt)
    length = np.fromfile(fobj, dt, 1)[0]
    np.fromfile(fobj, '|S%d' % length, 1)[0]
    entries_to_read = np.fromfile(fobj, dt, 1)[0]
    names = list()
    for _ in range(entries_to_read):
        idx = np.fromfile(fobj, dt, 1)[0]
        name_length = np.fromfile(fobj, dt, 1)[0]
        name = np.fromfile(fobj, '|S%d' % name_length, 1)[0]
        names.append(name)
        ctab[idx, :4] = np.fromfile(fobj, dt, 4)
    return (ctab, names)