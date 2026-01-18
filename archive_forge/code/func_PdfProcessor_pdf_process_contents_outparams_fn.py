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
def PdfProcessor_pdf_process_contents_outparams_fn(self, doc, res, stm, cookie):
    """
    Helper for out-params of class method pdf_processor::ll_pdf_process_contents() [pdf_process_contents()].
    """
    out_res = ll_pdf_process_contents(self.m_internal, doc.m_internal, res.m_internal, stm.m_internal, cookie.m_internal)
    return PdfObj(ll_pdf_keep_obj(out_res))