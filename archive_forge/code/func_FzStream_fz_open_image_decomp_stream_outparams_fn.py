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
def FzStream_fz_open_image_decomp_stream_outparams_fn(self, arg_1):
    """
    Helper for out-params of class method fz_stream::ll_fz_open_image_decomp_stream() [fz_open_image_decomp_stream()].
    """
    ret, l2factor = ll_fz_open_image_decomp_stream(self.m_internal, arg_1.m_internal)
    return (FzStream(ret), l2factor)