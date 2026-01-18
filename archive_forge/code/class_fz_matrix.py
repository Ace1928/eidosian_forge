from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class fz_matrix(object):
    """
    	fz_matrix is a row-major 3x3 matrix used for representing
    	transformations of coordinates throughout MuPDF.

    	Since all points reside in a two-dimensional space, one vector
    	is always a constant unit vector; hence only some elements may
    	vary in a matrix. Below is how the elements map between
    	different representations.

    a b 0
    	| c d 0 | normally represented as [ a b c d e f ].
    	\\ e f 1 /
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    a = property(_mupdf.fz_matrix_a_get, _mupdf.fz_matrix_a_set)
    b = property(_mupdf.fz_matrix_b_get, _mupdf.fz_matrix_b_set)
    c = property(_mupdf.fz_matrix_c_get, _mupdf.fz_matrix_c_set)
    d = property(_mupdf.fz_matrix_d_get, _mupdf.fz_matrix_d_set)
    e = property(_mupdf.fz_matrix_e_get, _mupdf.fz_matrix_e_set)
    f = property(_mupdf.fz_matrix_f_get, _mupdf.fz_matrix_f_set)

    def __init__(self):
        _mupdf.fz_matrix_swiginit(self, _mupdf.new_fz_matrix())
    __swig_destroy__ = _mupdf.delete_fz_matrix