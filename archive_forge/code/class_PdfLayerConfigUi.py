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
class PdfLayerConfigUi(object):
    """ Wrapper class for struct `pdf_layer_config_ui`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor sets .text to null, .type to PDF_LAYER_UI_LABEL, and other fields to zero.

        |

        *Overload 2:*
        We use default copy constructor and operator=.  Constructor using raw copy of pre-existing `::pdf_layer_config_ui`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_layer_config_ui`.
        """
        _mupdf.PdfLayerConfigUi_swiginit(self, _mupdf.new_PdfLayerConfigUi(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.PdfLayerConfigUi_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_PdfLayerConfigUi
    text = property(_mupdf.PdfLayerConfigUi_text_get, _mupdf.PdfLayerConfigUi_text_set)
    depth = property(_mupdf.PdfLayerConfigUi_depth_get, _mupdf.PdfLayerConfigUi_depth_set)
    type = property(_mupdf.PdfLayerConfigUi_type_get, _mupdf.PdfLayerConfigUi_type_set)
    selected = property(_mupdf.PdfLayerConfigUi_selected_get, _mupdf.PdfLayerConfigUi_selected_set)
    locked = property(_mupdf.PdfLayerConfigUi_locked_get, _mupdf.PdfLayerConfigUi_locked_set)
    s_num_instances = property(_mupdf.PdfLayerConfigUi_s_num_instances_get, _mupdf.PdfLayerConfigUi_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.PdfLayerConfigUi_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfLayerConfigUi___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfLayerConfigUi___ne__(self, rhs)