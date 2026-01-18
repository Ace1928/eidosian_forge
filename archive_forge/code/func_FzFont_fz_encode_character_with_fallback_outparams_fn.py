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
def FzFont_fz_encode_character_with_fallback_outparams_fn(self, unicode, script, language):
    """
    Helper for out-params of class method fz_font::ll_fz_encode_character_with_fallback() [fz_encode_character_with_fallback()].
    """
    ret, out_font = ll_fz_encode_character_with_fallback(self.m_internal, unicode, script, language)
    return (ret, FzFont(ll_fz_keep_font(out_font)))