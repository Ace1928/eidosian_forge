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
def PdfPage_pdf_page_presentation_outparams_fn(self, transition):
    """
    Helper for out-params of class method pdf_page::ll_pdf_page_presentation() [pdf_page_presentation()].
    """
    ret, duration = ll_pdf_page_presentation(self.m_internal, transition.internal())
    return (FzTransition(ret), duration)