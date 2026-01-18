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
def PdfAnnot_pdf_edit_text_field_value_outparams_fn(self, value, change):
    """
    Helper for out-params of class method pdf_annot::ll_pdf_edit_text_field_value() [pdf_edit_text_field_value()].
    """
    ret, selStart, selEnd, newvalue = ll_pdf_edit_text_field_value(self.m_internal, value, change)
    return (ret, selStart, selEnd, newvalue)