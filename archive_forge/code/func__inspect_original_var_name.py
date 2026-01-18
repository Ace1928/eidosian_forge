import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info
def _inspect_original_var_name(var, fallback_name):
    """
    Inspect variable name, will search above frames and fetch the same instance variable name
    in the most outer frame.
    If inspect failed, return fallback_name
    """
    if var is None:
        return fallback_name
    try:
        original_var_name = fallback_name
        frame = inspect.currentframe().f_back
        while frame is not None:
            arg_info = inspect.getargvalues(frame)
            fixed_args = [arg_info.locals[arg_name] for arg_name in arg_info.args]
            varlen_args = list(arg_info.locals[arg_info.varargs]) if arg_info.varargs else []
            keyword_args = list(arg_info.locals[arg_info.keywords].values()) if arg_info.keywords else []
            all_args = fixed_args + varlen_args + keyword_args
            if any((var is arg for arg in all_args)):
                frame = frame.f_back
                continue
            for var_name, var_val in frame.f_locals.items():
                if var_val is var:
                    original_var_name = var_name
                    break
            break
        return original_var_name
    except Exception:
        return fallback_name