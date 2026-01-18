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
def FzStextPage_fz_search_stext_page_outparams_fn(self, needle, hit_bbox, hit_max):
    """
    Helper for out-params of class method fz_stext_page::ll_fz_search_stext_page() [fz_search_stext_page()].
    """
    ret, hit_mark = ll_fz_search_stext_page(self.m_internal, needle, hit_bbox.internal(), hit_max)
    return (ret, hit_mark)