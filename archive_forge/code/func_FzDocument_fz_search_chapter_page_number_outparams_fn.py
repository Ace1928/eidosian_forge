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
def FzDocument_fz_search_chapter_page_number_outparams_fn(self, chapter, page, needle, hit_bbox, hit_max):
    """
    Helper for out-params of class method fz_document::ll_fz_search_chapter_page_number() [fz_search_chapter_page_number()].
    """
    ret, hit_mark = ll_fz_search_chapter_page_number(self.m_internal, chapter, page, needle, hit_bbox.internal(), hit_max)
    return (ret, hit_mark)