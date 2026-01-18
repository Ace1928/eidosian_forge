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
class FzStory(object):
    """ Wrapper class for struct `fz_story`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_draw_story(self, dev, ctm):
        """ Class-aware wrapper for `::fz_draw_story()`."""
        return _mupdf.FzStory_fz_draw_story(self, dev, ctm)

    def fz_place_story(self, where, filled):
        """ Class-aware wrapper for `::fz_place_story()`."""
        return _mupdf.FzStory_fz_place_story(self, where, filled)

    def fz_place_story_flags(self, where, filled, flags):
        """ Class-aware wrapper for `::fz_place_story_flags()`."""
        return _mupdf.FzStory_fz_place_story_flags(self, where, filled, flags)

    def fz_reset_story(self):
        """ Class-aware wrapper for `::fz_reset_story()`."""
        return _mupdf.FzStory_fz_reset_story(self)

    def fz_story_document(self):
        """ Class-aware wrapper for `::fz_story_document()`."""
        return _mupdf.FzStory_fz_story_document(self)

    def fz_story_positions(self, cb, arg):
        """ Class-aware wrapper for `::fz_story_positions()`."""
        return _mupdf.FzStory_fz_story_positions(self, cb, arg)

    def fz_story_warnings(self):
        """ Class-aware wrapper for `::fz_story_warnings()`."""
        return _mupdf.FzStory_fz_story_warnings(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `fz_new_story()`.

        |

        *Overload 2:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_story`.
        """
        _mupdf.FzStory_swiginit(self, _mupdf.new_FzStory(*args))
    __swig_destroy__ = _mupdf.delete_FzStory

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStory_m_internal_value(self)
    m_internal = property(_mupdf.FzStory_m_internal_get, _mupdf.FzStory_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStory_s_num_instances_get, _mupdf.FzStory_s_num_instances_set)