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
class FzOutlineItem(object):
    """ Wrapper class for struct `fz_outline_item`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def valid(self):
        return _mupdf.FzOutlineItem_valid(self)

    def title(self):
        return _mupdf.FzOutlineItem_title(self)

    def uri(self):
        return _mupdf.FzOutlineItem_uri(self)

    def is_open(self):
        return _mupdf.FzOutlineItem_is_open(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_outline_item`.
        """
        _mupdf.FzOutlineItem_swiginit(self, _mupdf.new_FzOutlineItem(*args))
    __swig_destroy__ = _mupdf.delete_FzOutlineItem
    s_num_instances = property(_mupdf.FzOutlineItem_s_num_instances_get, _mupdf.FzOutlineItem_s_num_instances_set)