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
def PdfCmap_pdf_decode_cmap_outparams_fn(self, s, e):
    """
    Helper for out-params of class method pdf_cmap::ll_pdf_decode_cmap() [pdf_decode_cmap()].
    """
    ret, cpt = ll_pdf_decode_cmap(self.m_internal, s, e)
    return (ret, cpt)