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
def PdfDocument_pdf_parse_ind_obj_outparams_fn(self, f):
    """
    Helper for out-params of class method pdf_document::ll_pdf_parse_ind_obj() [pdf_parse_ind_obj()].
    """
    ret, num, gen, stm_ofs, try_repair = ll_pdf_parse_ind_obj(self.m_internal, f.m_internal)
    return (PdfObj(ret), num, gen, stm_ofs, try_repair)