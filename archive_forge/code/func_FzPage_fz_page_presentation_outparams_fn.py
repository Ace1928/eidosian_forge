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
def FzPage_fz_page_presentation_outparams_fn(self, transition):
    """
    Helper for out-params of class method fz_page::ll_fz_page_presentation() [fz_page_presentation()].
    """
    ret, duration = ll_fz_page_presentation(self.m_internal, transition.internal())
    return (FzTransition(ret), duration)