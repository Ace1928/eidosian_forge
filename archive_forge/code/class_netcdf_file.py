import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
class netcdf_file:
    """
    A file object for NetCDF data.

    A `netcdf_file` object has two standard attributes: `dimensions` and
    `variables`. The values of both are dictionaries, mapping dimension
    names to their associated lengths and variable names to variables,
    respectively. Application programs should never modify these
    dictionaries.

    All other attributes correspond to global attributes defined in the
    NetCDF file. Global file attributes are created by assigning to an
    attribute of the `netcdf_file` object.

    Parameters
    ----------
    filename : string or file-like
        string -> filename
    mode : {'r', 'w', 'a'}, optional
        read-write-append mode, default is 'r'
    mmap : None or bool, optional
        Whether to mmap `filename` when reading.  Default is True
        when `filename` is a file name, False when `filename` is a
        file-like object. Note that when mmap is in use, data arrays
        returned refer directly to the mmapped data on disk, and the
        file cannot be closed as long as references to it exist.
    version : {1, 2}, optional
        version of netcdf to read / write, where 1 means *Classic
        format* and 2 means *64-bit offset format*.  Default is 1.  See
        `here <https://docs.unidata.ucar.edu/nug/current/netcdf_introduction.html#select_format>`__
        for more info.
    maskandscale : bool, optional
        Whether to automatically scale and/or mask data based on attributes.
        Default is False.

    Notes
    -----
    The major advantage of this module over other modules is that it doesn't
    require the code to be linked to the NetCDF libraries. This module is
    derived from `pupynere <https://bitbucket.org/robertodealmeida/pupynere/>`_.

    NetCDF files are a self-describing binary data format. The file contains
    metadata that describes the dimensions and variables in the file. More
    details about NetCDF files can be found `here
    <https://www.unidata.ucar.edu/software/netcdf/guide_toc.html>`__. There
    are three main sections to a NetCDF data structure:

    1. Dimensions
    2. Variables
    3. Attributes

    The dimensions section records the name and length of each dimension used
    by the variables. The variables would then indicate which dimensions it
    uses and any attributes such as data units, along with containing the data
    values for the variable. It is good practice to include a
    variable that is the same name as a dimension to provide the values for
    that axes. Lastly, the attributes section would contain additional
    information such as the name of the file creator or the instrument used to
    collect the data.

    When writing data to a NetCDF file, there is often the need to indicate the
    'record dimension'. A record dimension is the unbounded dimension for a
    variable. For example, a temperature variable may have dimensions of
    latitude, longitude and time. If one wants to add more temperature data to
    the NetCDF file as time progresses, then the temperature variable should
    have the time dimension flagged as the record dimension.

    In addition, the NetCDF file header contains the position of the data in
    the file, so access can be done in an efficient manner without loading
    unnecessary data into memory. It uses the ``mmap`` module to create
    Numpy arrays mapped to the data on disk, for the same purpose.

    Note that when `netcdf_file` is used to open a file with mmap=True
    (default for read-only), arrays returned by it refer to data
    directly on the disk. The file should not be closed, and cannot be cleanly
    closed when asked, if such arrays are alive. You may want to copy data arrays
    obtained from mmapped Netcdf file if they are to be processed after the file
    is closed, see the example below.

    Examples
    --------
    To create a NetCDF file:

    >>> from scipy.io import netcdf_file
    >>> import numpy as np
    >>> f = netcdf_file('simple.nc', 'w')
    >>> f.history = 'Created for a test'
    >>> f.createDimension('time', 10)
    >>> time = f.createVariable('time', 'i', ('time',))
    >>> time[:] = np.arange(10)
    >>> time.units = 'days since 2008-01-01'
    >>> f.close()

    Note the assignment of ``arange(10)`` to ``time[:]``.  Exposing the slice
    of the time variable allows for the data to be set in the object, rather
    than letting ``arange(10)`` overwrite the ``time`` variable.

    To read the NetCDF file we just created:

    >>> from scipy.io import netcdf_file
    >>> f = netcdf_file('simple.nc', 'r')
    >>> print(f.history)
    b'Created for a test'
    >>> time = f.variables['time']
    >>> print(time.units)
    b'days since 2008-01-01'
    >>> print(time.shape)
    (10,)
    >>> print(time[-1])
    9

    NetCDF files, when opened read-only, return arrays that refer
    directly to memory-mapped data on disk:

    >>> data = time[:]

    If the data is to be processed after the file is closed, it needs
    to be copied to main memory:

    >>> data = time[:].copy()
    >>> del time
    >>> f.close()
    >>> data.mean()
    4.5

    A NetCDF file can also be used as context manager:

    >>> from scipy.io import netcdf_file
    >>> with netcdf_file('simple.nc', 'r') as f:
    ...     print(f.history)
    b'Created for a test'

    """

    def __init__(self, filename, mode='r', mmap=None, version=1, maskandscale=False):
        """Initialize netcdf_file from fileobj (str or file-like)."""
        if mode not in 'rwa':
            raise ValueError("Mode must be either 'r', 'w' or 'a'.")
        if hasattr(filename, 'seek'):
            self.fp = filename
            self.filename = 'None'
            if mmap is None:
                mmap = False
            elif mmap and (not hasattr(filename, 'fileno')):
                raise ValueError('Cannot use file object for mmap')
        else:
            self.filename = filename
            omode = 'r+' if mode == 'a' else mode
            self.fp = open(self.filename, '%sb' % omode)
            if mmap is None:
                mmap = not IS_PYPY
        if mode != 'r':
            mmap = False
        self.use_mmap = mmap
        self.mode = mode
        self.version_byte = version
        self.maskandscale = maskandscale
        self.dimensions = {}
        self.variables = {}
        self._dims = []
        self._recs = 0
        self._recsize = 0
        self._mm = None
        self._mm_buf = None
        if self.use_mmap:
            self._mm = mm.mmap(self.fp.fileno(), 0, access=mm.ACCESS_READ)
            self._mm_buf = np.frombuffer(self._mm, dtype=np.int8)
        self._attributes = {}
        if mode in 'ra':
            self._read()

    def __setattr__(self, attr, value):
        try:
            self._attributes[attr] = value
        except AttributeError:
            pass
        self.__dict__[attr] = value

    def close(self):
        """Closes the NetCDF file."""
        if hasattr(self, 'fp') and (not self.fp.closed):
            try:
                self.flush()
            finally:
                self.variables = {}
                if self._mm_buf is not None:
                    ref = weakref.ref(self._mm_buf)
                    self._mm_buf = None
                    if ref() is None:
                        self._mm.close()
                    else:
                        warnings.warn('Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist. All data arrays obtained from such files refer directly to data on disk, and must be copied before the file can be cleanly closed. (See netcdf_file docstring for more information on mmap.)', category=RuntimeWarning, stacklevel=2)
                self._mm = None
                self.fp.close()
    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def createDimension(self, name, length):
        """
        Adds a dimension to the Dimension section of the NetCDF data structure.

        Note that this function merely adds a new dimension that the variables can
        reference. The values for the dimension, if desired, should be added as
        a variable using `createVariable`, referring to this dimension.

        Parameters
        ----------
        name : str
            Name of the dimension (Eg, 'lat' or 'time').
        length : int
            Length of the dimension.

        See Also
        --------
        createVariable

        """
        if length is None and self._dims:
            raise ValueError('Only first dimension may be unlimited!')
        self.dimensions[name] = length
        self._dims.append(name)

    def createVariable(self, name, type, dimensions):
        """
        Create an empty variable for the `netcdf_file` object, specifying its data
        type and the dimensions it uses.

        Parameters
        ----------
        name : str
            Name of the new variable.
        type : dtype or str
            Data type of the variable.
        dimensions : sequence of str
            List of the dimension names used by the variable, in the desired order.

        Returns
        -------
        variable : netcdf_variable
            The newly created ``netcdf_variable`` object.
            This object has also been added to the `netcdf_file` object as well.

        See Also
        --------
        createDimension

        Notes
        -----
        Any dimensions to be used by the variable should already exist in the
        NetCDF data structure or should be created by `createDimension` prior to
        creating the NetCDF variable.

        """
        shape = tuple([self.dimensions[dim] for dim in dimensions])
        shape_ = tuple([dim or 0 for dim in shape])
        type = dtype(type)
        typecode, size = (type.char, type.itemsize)
        if (typecode, size) not in REVERSE:
            raise ValueError('NetCDF 3 does not support type %s' % type)
        data = empty(shape_, dtype=type.newbyteorder('B'))
        self.variables[name] = netcdf_variable(data, typecode, size, shape, dimensions, maskandscale=self.maskandscale)
        return self.variables[name]

    def flush(self):
        """
        Perform a sync-to-disk flush if the `netcdf_file` object is in write mode.

        See Also
        --------
        sync : Identical function

        """
        if hasattr(self, 'mode') and self.mode in 'wa':
            self._write()
    sync = flush

    def _write(self):
        self.fp.seek(0)
        self.fp.write(b'CDF')
        self.fp.write(array(self.version_byte, '>b').tobytes())
        self._write_numrecs()
        self._write_dim_array()
        self._write_gatt_array()
        self._write_var_array()

    def _write_numrecs(self):
        for var in self.variables.values():
            if var.isrec and len(var.data) > self._recs:
                self.__dict__['_recs'] = len(var.data)
        self._pack_int(self._recs)

    def _write_dim_array(self):
        if self.dimensions:
            self.fp.write(NC_DIMENSION)
            self._pack_int(len(self.dimensions))
            for name in self._dims:
                self._pack_string(name)
                length = self.dimensions[name]
                self._pack_int(length or 0)
        else:
            self.fp.write(ABSENT)

    def _write_gatt_array(self):
        self._write_att_array(self._attributes)

    def _write_att_array(self, attributes):
        if attributes:
            self.fp.write(NC_ATTRIBUTE)
            self._pack_int(len(attributes))
            for name, values in attributes.items():
                self._pack_string(name)
                self._write_att_values(values)
        else:
            self.fp.write(ABSENT)

    def _write_var_array(self):
        if self.variables:
            self.fp.write(NC_VARIABLE)
            self._pack_int(len(self.variables))

            def sortkey(n):
                v = self.variables[n]
                if v.isrec:
                    return (-1,)
                return v._shape
            variables = sorted(self.variables, key=sortkey, reverse=True)
            for name in variables:
                self._write_var_metadata(name)
            self.__dict__['_recsize'] = sum([var._vsize for var in self.variables.values() if var.isrec])
            for name in variables:
                self._write_var_data(name)
        else:
            self.fp.write(ABSENT)

    def _write_var_metadata(self, name):
        var = self.variables[name]
        self._pack_string(name)
        self._pack_int(len(var.dimensions))
        for dimname in var.dimensions:
            dimid = self._dims.index(dimname)
            self._pack_int(dimid)
        self._write_att_array(var._attributes)
        nc_type = REVERSE[var.typecode(), var.itemsize()]
        self.fp.write(nc_type)
        if not var.isrec:
            vsize = var.data.size * var.data.itemsize
            vsize += -vsize % 4
        else:
            try:
                vsize = var.data[0].size * var.data.itemsize
            except IndexError:
                vsize = 0
            rec_vars = len([v for v in self.variables.values() if v.isrec])
            if rec_vars > 1:
                vsize += -vsize % 4
        self.variables[name].__dict__['_vsize'] = vsize
        self._pack_int(vsize)
        self.variables[name].__dict__['_begin'] = self.fp.tell()
        self._pack_begin(0)

    def _write_var_data(self, name):
        var = self.variables[name]
        the_beguine = self.fp.tell()
        self.fp.seek(var._begin)
        self._pack_begin(the_beguine)
        self.fp.seek(the_beguine)
        if not var.isrec:
            self.fp.write(var.data.tobytes())
            count = var.data.size * var.data.itemsize
            self._write_var_padding(var, var._vsize - count)
        else:
            if self._recs > len(var.data):
                shape = (self._recs,) + var.data.shape[1:]
                try:
                    var.data.resize(shape)
                except ValueError:
                    dtype = var.data.dtype
                    var.__dict__['data'] = np.resize(var.data, shape).astype(dtype)
            pos0 = pos = self.fp.tell()
            for rec in var.data:
                if not rec.shape and (rec.dtype.byteorder == '<' or (rec.dtype.byteorder == '=' and LITTLE_ENDIAN)):
                    rec = rec.byteswap()
                self.fp.write(rec.tobytes())
                count = rec.size * rec.itemsize
                self._write_var_padding(var, var._vsize - count)
                pos += self._recsize
                self.fp.seek(pos)
            self.fp.seek(pos0 + var._vsize)

    def _write_var_padding(self, var, size):
        encoded_fill_value = var._get_encoded_fill_value()
        num_fills = size // len(encoded_fill_value)
        self.fp.write(encoded_fill_value * num_fills)

    def _write_att_values(self, values):
        if hasattr(values, 'dtype'):
            nc_type = REVERSE[values.dtype.char, values.dtype.itemsize]
        else:
            types = [(int, NC_INT), (float, NC_FLOAT), (str, NC_CHAR)]
            if isinstance(values, (str, bytes)):
                sample = values
            else:
                try:
                    sample = values[0]
                except TypeError:
                    sample = values
            for class_, nc_type in types:
                if isinstance(sample, class_):
                    break
        typecode, size = TYPEMAP[nc_type]
        dtype_ = '>%s' % typecode
        dtype_ = 'S' if dtype_ == '>c' else dtype_
        values = asarray(values, dtype=dtype_)
        self.fp.write(nc_type)
        if values.dtype.char == 'S':
            nelems = values.itemsize
        else:
            nelems = values.size
        self._pack_int(nelems)
        if not values.shape and (values.dtype.byteorder == '<' or (values.dtype.byteorder == '=' and LITTLE_ENDIAN)):
            values = values.byteswap()
        self.fp.write(values.tobytes())
        count = values.size * values.itemsize
        self.fp.write(b'\x00' * (-count % 4))

    def _read(self):
        magic = self.fp.read(3)
        if not magic == b'CDF':
            raise TypeError('Error: %s is not a valid NetCDF 3 file' % self.filename)
        self.__dict__['version_byte'] = frombuffer(self.fp.read(1), '>b')[0]
        self._read_numrecs()
        self._read_dim_array()
        self._read_gatt_array()
        self._read_var_array()

    def _read_numrecs(self):
        self.__dict__['_recs'] = self._unpack_int()

    def _read_dim_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_DIMENSION]:
            raise ValueError('Unexpected header.')
        count = self._unpack_int()
        for dim in range(count):
            name = self._unpack_string().decode('latin1')
            length = self._unpack_int() or None
            self.dimensions[name] = length
            self._dims.append(name)

    def _read_gatt_array(self):
        for k, v in self._read_att_array().items():
            self.__setattr__(k, v)

    def _read_att_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_ATTRIBUTE]:
            raise ValueError('Unexpected header.')
        count = self._unpack_int()
        attributes = {}
        for attr in range(count):
            name = self._unpack_string().decode('latin1')
            attributes[name] = self._read_att_values()
        return attributes

    def _read_var_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_VARIABLE]:
            raise ValueError('Unexpected header.')
        begin = 0
        dtypes = {'names': [], 'formats': []}
        rec_vars = []
        count = self._unpack_int()
        for var in range(count):
            name, dimensions, shape, attributes, typecode, size, dtype_, begin_, vsize = self._read_var()
            if shape and shape[0] is None:
                rec_vars.append(name)
                self.__dict__['_recsize'] += vsize
                if begin == 0:
                    begin = begin_
                dtypes['names'].append(name)
                dtypes['formats'].append(str(shape[1:]) + dtype_)
                if typecode in 'bch':
                    actual_size = reduce(mul, (1,) + shape[1:]) * size
                    padding = -actual_size % 4
                    if padding:
                        dtypes['names'].append('_padding_%d' % var)
                        dtypes['formats'].append('(%d,)>b' % padding)
                data = None
            else:
                a_size = reduce(mul, shape, 1) * size
                if self.use_mmap:
                    data = self._mm_buf[begin_:begin_ + a_size].view(dtype=dtype_)
                    data.shape = shape
                else:
                    pos = self.fp.tell()
                    self.fp.seek(begin_)
                    data = frombuffer(self.fp.read(a_size), dtype=dtype_).copy()
                    data.shape = shape
                    self.fp.seek(pos)
            self.variables[name] = netcdf_variable(data, typecode, size, shape, dimensions, attributes, maskandscale=self.maskandscale)
        if rec_vars:
            if len(rec_vars) == 1:
                dtypes['names'] = dtypes['names'][:1]
                dtypes['formats'] = dtypes['formats'][:1]
            if self.use_mmap:
                buf = self._mm_buf[begin:begin + self._recs * self._recsize]
                rec_array = buf.view(dtype=dtypes)
                rec_array.shape = (self._recs,)
            else:
                pos = self.fp.tell()
                self.fp.seek(begin)
                rec_array = frombuffer(self.fp.read(self._recs * self._recsize), dtype=dtypes).copy()
                rec_array.shape = (self._recs,)
                self.fp.seek(pos)
            for var in rec_vars:
                self.variables[var].__dict__['data'] = rec_array[var]

    def _read_var(self):
        name = self._unpack_string().decode('latin1')
        dimensions = []
        shape = []
        dims = self._unpack_int()
        for i in range(dims):
            dimid = self._unpack_int()
            dimname = self._dims[dimid]
            dimensions.append(dimname)
            dim = self.dimensions[dimname]
            shape.append(dim)
        dimensions = tuple(dimensions)
        shape = tuple(shape)
        attributes = self._read_att_array()
        nc_type = self.fp.read(4)
        vsize = self._unpack_int()
        begin = [self._unpack_int, self._unpack_int64][self.version_byte - 1]()
        typecode, size = TYPEMAP[nc_type]
        dtype_ = '>%s' % typecode
        return (name, dimensions, shape, attributes, typecode, size, dtype_, begin, vsize)

    def _read_att_values(self):
        nc_type = self.fp.read(4)
        n = self._unpack_int()
        typecode, size = TYPEMAP[nc_type]
        count = n * size
        values = self.fp.read(int(count))
        self.fp.read(-count % 4)
        if typecode != 'c':
            values = frombuffer(values, dtype='>%s' % typecode).copy()
            if values.shape == (1,):
                values = values[0]
        else:
            values = values.rstrip(b'\x00')
        return values

    def _pack_begin(self, begin):
        if self.version_byte == 1:
            self._pack_int(begin)
        elif self.version_byte == 2:
            self._pack_int64(begin)

    def _pack_int(self, value):
        self.fp.write(array(value, '>i').tobytes())
    _pack_int32 = _pack_int

    def _unpack_int(self):
        return int(frombuffer(self.fp.read(4), '>i')[0])
    _unpack_int32 = _unpack_int

    def _pack_int64(self, value):
        self.fp.write(array(value, '>q').tobytes())

    def _unpack_int64(self):
        return frombuffer(self.fp.read(8), '>q')[0]

    def _pack_string(self, s):
        count = len(s)
        self._pack_int(count)
        self.fp.write(s.encode('latin1'))
        self.fp.write(b'\x00' * (-count % 4))

    def _unpack_string(self):
        count = self._unpack_int()
        s = self.fp.read(count).rstrip(b'\x00')
        self.fp.read(-count % 4)
        return s