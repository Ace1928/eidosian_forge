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
def PdfObj_pdf_array_get_string_outparams_fn(self, index):
    """
    Helper for out-params of class method pdf_obj::ll_pdf_array_get_string() [pdf_array_get_string()].
    """
    ret, sizep = ll_pdf_array_get_string(self.m_internal, index)
    return (ret, sizep)