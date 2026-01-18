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
def FzImage_fz_get_pixmap_from_image_outparams_fn(self, subarea, ctm):
    """
    Helper for out-params of class method fz_image::ll_fz_get_pixmap_from_image() [fz_get_pixmap_from_image()].
    """
    ret, w, h = ll_fz_get_pixmap_from_image(self.m_internal, subarea.internal(), ctm.internal())
    return (FzPixmap(ret), w, h)