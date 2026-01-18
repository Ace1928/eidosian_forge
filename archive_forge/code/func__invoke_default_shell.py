import os
import sys
from importlib import import_module
import click
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
def _invoke_default_shell(locals):
    try:
        import IPython
    except ImportError:
        try:
            import bpython
        except ImportError:
            _invoke_fallback_shell(locals)
        else:
            _invoke_bpython_shell(locals)
    else:
        _invoke_ipython_shell(locals)