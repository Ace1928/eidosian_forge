from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class HeapTypeObjectPtr(PyObjectPtr):
    _typename = 'PyObject'

    def get_attr_dict(self):
        """
        Get the PyDictObject ptr representing the attribute dictionary
        (or None if there's a problem)
        """
        try:
            typeobj = self.type()
            dictoffset = int_from_int(typeobj.field('tp_dictoffset'))
            if dictoffset != 0:
                if dictoffset < 0:
                    type_PyVarObject_ptr = gdb.lookup_type('PyVarObject').pointer()
                    tsize = int_from_int(self._gdbval.cast(type_PyVarObject_ptr)['ob_size'])
                    if tsize < 0:
                        tsize = -tsize
                    size = _PyObject_VAR_SIZE(typeobj, tsize)
                    dictoffset += size
                    assert dictoffset > 0
                    assert dictoffset % _sizeof_void_p() == 0
                dictptr = self._gdbval.cast(_type_char_ptr()) + dictoffset
                PyObjectPtrPtr = PyObjectPtr.get_gdb_type().pointer()
                dictptr = dictptr.cast(PyObjectPtrPtr)
                return PyObjectPtr.from_pyobject_ptr(dictptr.dereference())
        except RuntimeError:
            pass
        return None

    def proxyval(self, visited):
        """
        Support for classes.

        Currently we just locate the dictionary using a transliteration to
        python of _PyObject_GetDictPtr, ignoring descriptors
        """
        if self.as_address() in visited:
            return ProxyAlreadyVisited('<...>')
        visited.add(self.as_address())
        pyop_attr_dict = self.get_attr_dict()
        if pyop_attr_dict:
            attr_dict = pyop_attr_dict.proxyval(visited)
        else:
            attr_dict = {}
        tp_name = self.safe_tp_name()
        return InstanceProxy(tp_name, attr_dict, long(self._gdbval))

    def write_repr(self, out, visited):
        if self.as_address() in visited:
            out.write('<...>')
            return
        visited.add(self.as_address())
        pyop_attrdict = self.get_attr_dict()
        _write_instance_repr(out, visited, self.safe_tp_name(), pyop_attrdict, self.as_address())