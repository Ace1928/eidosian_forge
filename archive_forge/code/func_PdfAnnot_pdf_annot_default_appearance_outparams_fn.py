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
def PdfAnnot_pdf_annot_default_appearance_outparams_fn(self, color):
    """
    Helper for out-params of class method pdf_annot::ll_pdf_annot_default_appearance() [pdf_annot_default_appearance()].
    """
    font, size, n = ll_pdf_annot_default_appearance(self.m_internal, color)
    return (font, size, n)