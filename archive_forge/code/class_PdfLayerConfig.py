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
class PdfLayerConfig(object):
    """ Wrapper class for struct `pdf_layer_config`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_layer_config`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_layer_config`.
        """
        _mupdf.PdfLayerConfig_swiginit(self, _mupdf.new_PdfLayerConfig(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.PdfLayerConfig_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_PdfLayerConfig
    name = property(_mupdf.PdfLayerConfig_name_get, _mupdf.PdfLayerConfig_name_set)
    creator = property(_mupdf.PdfLayerConfig_creator_get, _mupdf.PdfLayerConfig_creator_set)
    s_num_instances = property(_mupdf.PdfLayerConfig_s_num_instances_get, _mupdf.PdfLayerConfig_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.PdfLayerConfig_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfLayerConfig___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfLayerConfig___ne__(self, rhs)