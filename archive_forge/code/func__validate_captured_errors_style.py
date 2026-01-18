import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def _validate_captured_errors_style(style_str):
    from numba.core.errors import NumbaPendingDeprecationWarning
    rendered_style = str(style_str)
    if rendered_style not in ('new_style', 'old_style', 'default'):
        msg = f'Invalid style in NUMBA_CAPTURED_ERRORS: {rendered_style}'
        raise ValueError(msg)
    else:
        if rendered_style == 'default':
            rendered_style = 'old_style'
        elif rendered_style == 'old_style':
            warnings.warn(_old_style_deprecation_msg, NumbaPendingDeprecationWarning)
        return rendered_style