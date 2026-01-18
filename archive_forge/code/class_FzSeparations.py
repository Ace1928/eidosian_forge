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
class FzSeparations(object):
    """ Wrapper class for struct `fz_separations`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_add_separation(self, name, cs, cs_channel):
        """
        Class-aware wrapper for `::fz_add_separation()`.
        	Add a separation (null terminated name, colorspace)
        """
        return _mupdf.FzSeparations_fz_add_separation(self, name, cs, cs_channel)

    def fz_add_separation_equivalents(self, rgba, cmyk, name):
        """
        Class-aware wrapper for `::fz_add_separation_equivalents()`.
        	Add a separation with equivalents (null terminated name,
        	colorspace)

        	(old, deprecated)
        """
        return _mupdf.FzSeparations_fz_add_separation_equivalents(self, rgba, cmyk, name)

    def fz_clone_separations_for_overprint(self):
        """
        Class-aware wrapper for `::fz_clone_separations_for_overprint()`.
        	Return a separations object with all the spots in the input
        	separations object that are set to composite, reset to be
        	enabled. If there ARE no spots in the object, this returns
        	NULL. If the object already has all its spots enabled, then
        	just returns another handle on the same object.
        """
        return _mupdf.FzSeparations_fz_clone_separations_for_overprint(self)

    def fz_compare_separations(self, sep2):
        """
        Class-aware wrapper for `::fz_compare_separations()`.
        	Compare 2 separations structures (or NULLs).

        	Return 0 if identical, non-zero if not identical.
        """
        return _mupdf.FzSeparations_fz_compare_separations(self, sep2)

    def fz_count_active_separations(self):
        """
        Class-aware wrapper for `::fz_count_active_separations()`.
        	Return the number of active separations.
        """
        return _mupdf.FzSeparations_fz_count_active_separations(self)

    def fz_count_separations(self):
        """ Class-aware wrapper for `::fz_count_separations()`."""
        return _mupdf.FzSeparations_fz_count_separations(self)

    def fz_separation_equivalent(self, idx, dst_cs, dst_color, prf, color_params):
        """
        Class-aware wrapper for `::fz_separation_equivalent()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_separation_equivalent(int idx, ::fz_colorspace *dst_cs, ::fz_colorspace *prf, ::fz_color_params color_params)` => float dst_color

        	Get the equivalent separation color in a given colorspace.
        """
        return _mupdf.FzSeparations_fz_separation_equivalent(self, idx, dst_cs, dst_color, prf, color_params)

    def fz_separation_name(self, separation):
        """ Class-aware wrapper for `::fz_separation_name()`."""
        return _mupdf.FzSeparations_fz_separation_name(self, separation)

    def fz_set_separation_behavior(self, separation, behavior):
        """
        Class-aware wrapper for `::fz_set_separation_behavior()`.
        	Control the rendering of a given separation.
        """
        return _mupdf.FzSeparations_fz_set_separation_behavior(self, separation, behavior)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_separations()`.
        		Create a new separations structure (initially empty)


        |

        *Overload 2:*
         Copy constructor using `fz_keep_separations()`.

        |

        *Overload 3:*
         Constructor using raw copy of pre-existing `::fz_separations`.

        |

        *Overload 4:*
         Constructor using raw copy of pre-existing `::fz_separations`.
        """
        _mupdf.FzSeparations_swiginit(self, _mupdf.new_FzSeparations(*args))
    __swig_destroy__ = _mupdf.delete_FzSeparations

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzSeparations_m_internal_value(self)
    m_internal = property(_mupdf.FzSeparations_m_internal_get, _mupdf.FzSeparations_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzSeparations_s_num_instances_get, _mupdf.FzSeparations_s_num_instances_set)