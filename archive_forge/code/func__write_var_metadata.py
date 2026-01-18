import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
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