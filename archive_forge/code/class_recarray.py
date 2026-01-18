import warnings
from collections import Counter
from contextlib import nullcontext
from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from numpy.compat import os_fspath
from .arrayprint import _get_legacy_print_mode
class recarray(ndarray):
    """Construct an ndarray that allows field access using attributes.

    Arrays may have a data-types containing fields, analogous
    to columns in a spread sheet.  An example is ``[(x, int), (y, float)]``,
    where each entry in the array is a pair of ``(int, float)``.  Normally,
    these attributes are accessed using dictionary lookups such as ``arr['x']``
    and ``arr['y']``.  Record arrays allow the fields to be accessed as members
    of the array, using ``arr.x`` and ``arr.y``.

    Parameters
    ----------
    shape : tuple
        Shape of output array.
    dtype : data-type, optional
        The desired data-type.  By default, the data-type is determined
        from `formats`, `names`, `titles`, `aligned` and `byteorder`.
    formats : list of data-types, optional
        A list containing the data-types for the different columns, e.g.
        ``['i4', 'f8', 'i4']``.  `formats` does *not* support the new
        convention of using types directly, i.e. ``(int, float, int)``.
        Note that `formats` must be a list, not a tuple.
        Given that `formats` is somewhat limited, we recommend specifying
        `dtype` instead.
    names : tuple of str, optional
        The name of each column, e.g. ``('x', 'y', 'z')``.
    buf : buffer, optional
        By default, a new array is created of the given shape and data-type.
        If `buf` is specified and is an object exposing the buffer interface,
        the array will use the memory from the existing buffer.  In this case,
        the `offset` and `strides` keywords are available.

    Other Parameters
    ----------------
    titles : tuple of str, optional
        Aliases for column names.  For example, if `names` were
        ``('x', 'y', 'z')`` and `titles` is
        ``('x_coordinate', 'y_coordinate', 'z_coordinate')``, then
        ``arr['x']`` is equivalent to both ``arr.x`` and ``arr.x_coordinate``.
    byteorder : {'<', '>', '='}, optional
        Byte-order for all fields.
    aligned : bool, optional
        Align the fields in memory as the C-compiler would.
    strides : tuple of ints, optional
        Buffer (`buf`) is interpreted according to these strides (strides
        define how many bytes each array element, row, column, etc.
        occupy in memory).
    offset : int, optional
        Start reading buffer (`buf`) from this offset onwards.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    rec : recarray
        Empty array of the given shape and type.

    See Also
    --------
    core.records.fromrecords : Construct a record array from data.
    record : fundamental data-type for `recarray`.
    format_parser : determine a data-type from formats, names, titles.

    Notes
    -----
    This constructor can be compared to ``empty``: it creates a new record
    array but does not fill it with data.  To create a record array from data,
    use one of the following methods:

    1. Create a standard ndarray and convert it to a record array,
       using ``arr.view(np.recarray)``
    2. Use the `buf` keyword.
    3. Use `np.rec.fromrecords`.

    Examples
    --------
    Create an array with two fields, ``x`` and ``y``:

    >>> x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', '<f8'), ('y', '<i8')])
    >>> x
    array([(1., 2), (3., 4)], dtype=[('x', '<f8'), ('y', '<i8')])

    >>> x['x']
    array([1., 3.])

    View the array as a record array:

    >>> x = x.view(np.recarray)

    >>> x.x
    array([1., 3.])

    >>> x.y
    array([2, 4])

    Create a new, empty record array:

    >>> np.recarray((2,),
    ... dtype=[('x', int), ('y', float), ('z', int)]) #doctest: +SKIP
    rec.array([(-1073741821, 1.2249118382103472e-301, 24547520),
           (3471280, 1.2134086255804012e-316, 0)],
          dtype=[('x', '<i4'), ('y', '<f8'), ('z', '<i4')])

    """
    __name__ = 'recarray'
    __module__ = 'numpy'

    def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None, formats=None, names=None, titles=None, byteorder=None, aligned=False, order='C'):
        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(formats, names, titles, aligned, byteorder).dtype
        if buf is None:
            self = ndarray.__new__(subtype, shape, (record, descr), order=order)
        else:
            self = ndarray.__new__(subtype, shape, (record, descr), buffer=buf, offset=offset, strides=strides, order=order)
        return self

    def __array_finalize__(self, obj):
        if self.dtype.type is not record and self.dtype.names is not None:
            self.dtype = self.dtype

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError('recarray has no attribute %s' % attr) from e
        obj = self.getfield(*res)
        if obj.dtype.names is not None:
            if issubclass(obj.dtype.type, nt.void):
                return obj.view(dtype=(self.dtype.type, obj.dtype))
            return obj
        else:
            return obj.view(ndarray)

    def __setattr__(self, attr, val):
        if attr == 'dtype' and issubclass(val.type, nt.void) and (val.names is not None):
            val = sb.dtype((record, val))
        newattr = attr not in self.__dict__
        try:
            ret = object.__setattr__(self, attr, val)
        except Exception:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                raise
        else:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                return ret
            if newattr:
                try:
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError('record array has no attribute %s' % attr) from e
        return self.setfield(val, *res)

    def __getitem__(self, indx):
        obj = super().__getitem__(indx)
        if isinstance(obj, ndarray):
            if obj.dtype.names is not None:
                obj = obj.view(type(self))
                if issubclass(obj.dtype.type, nt.void):
                    return obj.view(dtype=(self.dtype.type, obj.dtype))
                return obj
            else:
                return obj.view(type=ndarray)
        else:
            return obj

    def __repr__(self):
        repr_dtype = self.dtype
        if self.dtype.type is record or not issubclass(self.dtype.type, nt.void):
            if repr_dtype.type is record:
                repr_dtype = sb.dtype((nt.void, repr_dtype))
            prefix = 'rec.array('
            fmt = 'rec.array(%s,%sdtype=%s)'
        else:
            prefix = 'array('
            fmt = 'array(%s,%sdtype=%s).view(numpy.recarray)'
        if self.size > 0 or self.shape == (0,):
            lst = sb.array2string(self, separator=', ', prefix=prefix, suffix=',')
        else:
            lst = '[], shape=%s' % (repr(self.shape),)
        lf = '\n' + ' ' * len(prefix)
        if _get_legacy_print_mode() <= 113:
            lf = ' ' + lf
        return fmt % (lst, lf, repr_dtype)

    def field(self, attr, val=None):
        if isinstance(attr, int):
            names = ndarray.__getattribute__(self, 'dtype').names
            attr = names[attr]
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        res = fielddict[attr][:2]
        if val is None:
            obj = self.getfield(*res)
            if obj.dtype.names is not None:
                return obj
            return obj.view(ndarray)
        else:
            return self.setfield(val, *res)