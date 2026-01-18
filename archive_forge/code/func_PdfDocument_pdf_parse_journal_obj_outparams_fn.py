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
def PdfDocument_pdf_parse_journal_obj_outparams_fn(self, stm):
    """
    Helper for out-params of class method pdf_document::ll_pdf_parse_journal_obj() [pdf_parse_journal_obj()].
    """
    ret, onum, ostm, newobj = ll_pdf_parse_journal_obj(self.m_internal, stm.m_internal)
    return (PdfObj(ret), onum, FzBuffer(ostm), newobj)