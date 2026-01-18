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
def FzOutput_fz_new_svg_device_with_id_outparams_fn(self, page_width, page_height, text_format, reuse_images):
    """
    Helper for out-params of class method fz_output::ll_fz_new_svg_device_with_id() [fz_new_svg_device_with_id()].
    """
    ret, id = ll_fz_new_svg_device_with_id(self.m_internal, page_width, page_height, text_format, reuse_images)
    return (FzDevice(ret), id)