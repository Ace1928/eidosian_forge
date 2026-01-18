import os
import sys
from importlib import import_module
import click
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
def _ipython_010(locals):
    from IPython.Shell import IPShell
    IPShell(argv=[], user_ns=locals).mainloop()