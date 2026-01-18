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
def PdfDocument_pdf_field_event_validate_outparams_fn(self, field, value):
    """
    Helper for out-params of class method pdf_document::ll_pdf_field_event_validate() [pdf_field_event_validate()].
    """
    ret, newvalue = ll_pdf_field_event_validate(self.m_internal, field.m_internal, value)
    return (ret, newvalue)