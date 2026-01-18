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
class FzMatrix(object):
    """
     Wrapper class for struct `fz_matrix`.
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

    @staticmethod
    def fz_scale(sx, sy):
        """
        Class-aware wrapper for `::fz_scale()`.
        	Create a scaling matrix.

        	The returned matrix is of the form [ sx 0 0 sy 0 0 ].

        	m: Pointer to the matrix to populate

        	sx, sy: Scaling factors along the X- and Y-axes. A scaling
        	factor of 1.0 will not cause any scaling along the relevant
        	axis.

        	Returns m.
        """
        return _mupdf.FzMatrix_fz_scale(sx, sy)

    @staticmethod
    def fz_shear(sx, sy):
        """
        Class-aware wrapper for `::fz_shear()`.
        	Create a shearing matrix.

        	The returned matrix is of the form [ 1 sy sx 1 0 0 ].

        	m: pointer to place to store returned matrix

        	sx, sy: Shearing factors. A shearing factor of 0.0 will not
        	cause any shearing along the relevant axis.

        	Returns m.
        """
        return _mupdf.FzMatrix_fz_shear(sx, sy)

    @staticmethod
    def fz_rotate(degrees):
        """
        Class-aware wrapper for `::fz_rotate()`.
        	Create a rotation matrix.

        	The returned matrix is of the form
        	[ cos(deg) sin(deg) -sin(deg) cos(deg) 0 0 ].

        	m: Pointer to place to store matrix

        	degrees: Degrees of counter clockwise rotation. Values less
        	than zero and greater than 360 are handled as expected.

        	Returns m.
        """
        return _mupdf.FzMatrix_fz_rotate(degrees)

    @staticmethod
    def fz_translate(tx, ty):
        """
        Class-aware wrapper for `::fz_translate()`.
        	Create a translation matrix.

        	The returned matrix is of the form [ 1 0 0 1 tx ty ].

        	m: A place to store the created matrix.

        	tx, ty: Translation distances along the X- and Y-axes. A
        	translation of 0 will not cause any translation along the
        	relevant axis.

        	Returns m.
        """
        return _mupdf.FzMatrix_fz_translate(tx, ty)

    @staticmethod
    def fz_transform_page(mediabox, resolution, rotate):
        """
        Class-aware wrapper for `::fz_transform_page()`.
        	Create transform matrix to draw page
        	at a given resolution and rotation. Adjusts the scaling
        	factors so that the page covers whole number of
        	pixels and adjust the page origin to be at 0,0.
        """
        return _mupdf.FzMatrix_fz_transform_page(mediabox, resolution, rotate)

    def fz_concat(self, *args):
        """
        *Overload 1:*
         We use default copy constructor and operator=.  Class-aware wrapper for `::fz_concat()`.
        		Multiply two matrices.

        		The order of the two matrices are important since matrix
        		multiplication is not commutative.

        		Returns result.


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_concat()`.
        		Multiply two matrices.

        		The order of the two matrices are important since matrix
        		multiplication is not commutative.

        		Returns result.
        """
        return _mupdf.FzMatrix_fz_concat(self, *args)

    def fz_invert_matrix(self):
        """
        Class-aware wrapper for `::fz_invert_matrix()`.
        	Create an inverse matrix.

        	inverse: Place to store inverse matrix.

        	matrix: Matrix to invert. A degenerate matrix, where the
        	determinant is equal to zero, can not be inverted and the
        	original matrix is returned instead.

        	Returns inverse.
        """
        return _mupdf.FzMatrix_fz_invert_matrix(self)

    def fz_is_identity(self):
        """ Class-aware wrapper for `::fz_is_identity()`."""
        return _mupdf.FzMatrix_fz_is_identity(self)

    def fz_is_rectilinear(self):
        """
        Class-aware wrapper for `::fz_is_rectilinear()`.
        	Check if a transformation is rectilinear.

        	Rectilinear means that no shearing is present and that any
        	rotations present are a multiple of 90 degrees. Usually this
        	is used to make sure that axis-aligned rectangles before the
        	transformation are still axis-aligned rectangles afterwards.
        """
        return _mupdf.FzMatrix_fz_is_rectilinear(self)

    def fz_matrix_expansion(self):
        """
        Class-aware wrapper for `::fz_matrix_expansion()`.
        	Calculate average scaling factor of matrix.
        """
        return _mupdf.FzMatrix_fz_matrix_expansion(self)

    def fz_matrix_max_expansion(self):
        """
        Class-aware wrapper for `::fz_matrix_max_expansion()`.
        	Find the largest expansion performed by this matrix.
        	(i.e. max(abs(m.a),abs(m.b),abs(m.c),abs(m.d))
        """
        return _mupdf.FzMatrix_fz_matrix_max_expansion(self)

    def fz_post_scale(self, sx, sy):
        """
        Class-aware wrapper for `::fz_post_scale()`.
        	Scale a matrix by postmultiplication.

        	m: Pointer to the matrix to scale

        	sx, sy: Scaling factors along the X- and Y-axes. A scaling
        	factor of 1.0 will not cause any scaling along the relevant
        	axis.

        	Returns m (updated).
        """
        return _mupdf.FzMatrix_fz_post_scale(self, sx, sy)

    def fz_pre_rotate(self, degrees):
        """
        Class-aware wrapper for `::fz_pre_rotate()`.
        	Rotate a transformation by premultiplying.

        	The premultiplied matrix is of the form
        	[ cos(deg) sin(deg) -sin(deg) cos(deg) 0 0 ].

        	m: Pointer to matrix to premultiply.

        	degrees: Degrees of counter clockwise rotation. Values less
        	than zero and greater than 360 are handled as expected.

        	Returns m (updated).
        """
        return _mupdf.FzMatrix_fz_pre_rotate(self, degrees)

    def fz_pre_scale(self, sx, sy):
        """
        Class-aware wrapper for `::fz_pre_scale()`.
        	Scale a matrix by premultiplication.

        	m: Pointer to the matrix to scale

        	sx, sy: Scaling factors along the X- and Y-axes. A scaling
        	factor of 1.0 will not cause any scaling along the relevant
        	axis.

        	Returns m (updated).
        """
        return _mupdf.FzMatrix_fz_pre_scale(self, sx, sy)

    def fz_pre_shear(self, sx, sy):
        """
        Class-aware wrapper for `::fz_pre_shear()`.
        	Premultiply a matrix with a shearing matrix.

        	The shearing matrix is of the form [ 1 sy sx 1 0 0 ].

        	m: pointer to matrix to premultiply

        	sx, sy: Shearing factors. A shearing factor of 0.0 will not
        	cause any shearing along the relevant axis.

        	Returns m (updated).
        """
        return _mupdf.FzMatrix_fz_pre_shear(self, sx, sy)

    def fz_pre_translate(self, tx, ty):
        """
        Class-aware wrapper for `::fz_pre_translate()`.
        	Translate a matrix by premultiplication.

        	m: The matrix to translate

        	tx, ty: Translation distances along the X- and Y-axes. A
        	translation of 0 will not cause any translation along the
        	relevant axis.

        	Returns m.
        """
        return _mupdf.FzMatrix_fz_pre_translate(self, tx, ty)

    def fz_subpixel_adjust(self, subpix_ctm, qe, qf):
        """
        Class-aware wrapper for `::fz_subpixel_adjust()`.
        	Perform subpixel quantisation and adjustment on a glyph matrix.

        	ctm: On entry, the desired 'ideal' transformation for a glyph.
        	On exit, adjusted to a (very similar) transformation quantised
        	for subpixel caching.

        	subpix_ctm: Initialised by the routine to the transform that
        	should be used to render the glyph.

        	qe, qf: which subpixel position we quantised to.

        	Returns: the size of the glyph.

        	Note: This is currently only exposed for use in our app. It
        	should be considered "at risk" of removal from the API.
        """
        return _mupdf.FzMatrix_fz_subpixel_adjust(self, subpix_ctm, qe, qf)

    def fz_try_invert_matrix(self, src):
        """
        Class-aware wrapper for `::fz_try_invert_matrix()`.
        	Attempt to create an inverse matrix.

        	inverse: Place to store inverse matrix.

        	matrix: Matrix to invert. A degenerate matrix, where the
        	determinant is equal to zero, can not be inverted.

        	Returns 1 if matrix is degenerate (singular), or 0 otherwise.
        """
        return _mupdf.FzMatrix_fz_try_invert_matrix(self, src)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `fz_make_matrix()`.

        |

        *Overload 2:*
        Constructs identity matrix (like fz_identity).

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_matrix`.

        |

        *Overload 4:*
        Constructor using raw copy of pre-existing `::fz_matrix`.
        """
        _mupdf.FzMatrix_swiginit(self, _mupdf.new_FzMatrix(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzMatrix_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzMatrix
    a = property(_mupdf.FzMatrix_a_get, _mupdf.FzMatrix_a_set)
    b = property(_mupdf.FzMatrix_b_get, _mupdf.FzMatrix_b_set)
    c = property(_mupdf.FzMatrix_c_get, _mupdf.FzMatrix_c_set)
    d = property(_mupdf.FzMatrix_d_get, _mupdf.FzMatrix_d_set)
    e = property(_mupdf.FzMatrix_e_get, _mupdf.FzMatrix_e_set)
    f = property(_mupdf.FzMatrix_f_get, _mupdf.FzMatrix_f_set)
    s_num_instances = property(_mupdf.FzMatrix_s_num_instances_get, _mupdf.FzMatrix_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzMatrix_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzMatrix___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzMatrix___ne__(self, rhs)