import warnings
from collections import Counter
from contextlib import nullcontext
from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from numpy.compat import os_fspath
from .arrayprint import _get_legacy_print_mode
@set_module('numpy')
class format_parser:
    """
    Class to convert formats, names, titles description to a dtype.

    After constructing the format_parser object, the dtype attribute is
    the converted data-type:
    ``dtype = format_parser(formats, names, titles).dtype``

    Attributes
    ----------
    dtype : dtype
        The converted data-type.

    Parameters
    ----------
    formats : str or list of str
        The format description, either specified as a string with
        comma-separated format descriptions in the form ``'f8, i4, a5'``, or
        a list of format description strings  in the form
        ``['f8', 'i4', 'a5']``.
    names : str or list/tuple of str
        The field names, either specified as a comma-separated string in the
        form ``'col1, col2, col3'``, or as a list or tuple of strings in the
        form ``['col1', 'col2', 'col3']``.
        An empty list can be used, in that case default field names
        ('f0', 'f1', ...) are used.
    titles : sequence
        Sequence of title strings. An empty list can be used to leave titles
        out.
    aligned : bool, optional
        If True, align the fields by padding as the C-compiler would.
        Default is False.
    byteorder : str, optional
        If specified, all the fields will be changed to the
        provided byte-order.  Otherwise, the default byte-order is
        used. For all available string specifiers, see `dtype.newbyteorder`.

    See Also
    --------
    dtype, typename, sctype2char

    Examples
    --------
    >>> np.format_parser(['<f8', '<i4', '<a5'], ['col1', 'col2', 'col3'],
    ...                  ['T1', 'T2', 'T3']).dtype
    dtype([(('T1', 'col1'), '<f8'), (('T2', 'col2'), '<i4'), (('T3', 'col3'), 'S5')])

    `names` and/or `titles` can be empty lists. If `titles` is an empty list,
    titles will simply not appear. If `names` is empty, default field names
    will be used.

    >>> np.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],
    ...                  []).dtype
    dtype([('col1', '<f8'), ('col2', '<i4'), ('col3', '<S5')])
    >>> np.format_parser(['<f8', '<i4', '<a5'], [], []).dtype
    dtype([('f0', '<f8'), ('f1', '<i4'), ('f2', 'S5')])

    """

    def __init__(self, formats, names, titles, aligned=False, byteorder=None):
        self._parseFormats(formats, aligned)
        self._setfieldnames(names, titles)
        self._createdtype(byteorder)

    def _parseFormats(self, formats, aligned=False):
        """ Parse the field formats """
        if formats is None:
            raise ValueError('Need formats argument')
        if isinstance(formats, list):
            dtype = sb.dtype([('f{}'.format(i), format_) for i, format_ in enumerate(formats)], aligned)
        else:
            dtype = sb.dtype(formats, aligned)
        fields = dtype.fields
        if fields is None:
            dtype = sb.dtype([('f1', dtype)], aligned)
            fields = dtype.fields
        keys = dtype.names
        self._f_formats = [fields[key][0] for key in keys]
        self._offsets = [fields[key][1] for key in keys]
        self._nfields = len(keys)

    def _setfieldnames(self, names, titles):
        """convert input field names into a list and assign to the _names
        attribute """
        if names:
            if type(names) in [list, tuple]:
                pass
            elif isinstance(names, str):
                names = names.split(',')
            else:
                raise NameError('illegal input names %s' % repr(names))
            self._names = [n.strip() for n in names[:self._nfields]]
        else:
            self._names = []
        self._names += ['f%d' % i for i in range(len(self._names), self._nfields)]
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError('Duplicate field names: %s' % _dup)
        if titles:
            self._titles = [n.strip() for n in titles[:self._nfields]]
        else:
            self._titles = []
            titles = []
        if self._nfields > len(titles):
            self._titles += [None] * (self._nfields - len(titles))

    def _createdtype(self, byteorder):
        dtype = sb.dtype({'names': self._names, 'formats': self._f_formats, 'offsets': self._offsets, 'titles': self._titles})
        if byteorder is not None:
            byteorder = _byteorderconv[byteorder[0]]
            dtype = dtype.newbyteorder(byteorder)
        self.dtype = dtype