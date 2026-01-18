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
def PdfDocument_pdf_count_q_balance_outparams_fn(self, res, stm):
    """
    Helper for out-params of class method pdf_document::ll_pdf_count_q_balance() [pdf_count_q_balance()].
    """
    underflow, overflow = ll_pdf_count_q_balance(self.m_internal, res.m_internal, stm.m_internal)
    return (underflow, overflow)