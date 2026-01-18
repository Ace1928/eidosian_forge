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
def FzBitmap_fz_bitmap_details_outparams_fn(self):
    """
    Helper for out-params of class method fz_bitmap::ll_fz_bitmap_details() [fz_bitmap_details()].
    """
    w, h, n, stride = ll_fz_bitmap_details(self.m_internal)
    return (w, h, n, stride)