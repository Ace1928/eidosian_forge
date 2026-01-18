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
def FzBuffer_fz_buffer_extract_outparams_fn(self):
    """
    Helper for out-params of class method fz_buffer::ll_fz_buffer_extract() [fz_buffer_extract()].
    """
    ret, data = ll_fz_buffer_extract(self.m_internal)
    return (ret, data)