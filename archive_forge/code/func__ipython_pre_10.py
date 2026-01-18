import os
import sys
from importlib import import_module
import click
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
def _ipython_pre_10(locals):
    from IPython.frontend.terminal.ipapp import TerminalIPythonApp
    app = TerminalIPythonApp.instance()
    app.initialize(argv=[])
    app.shell.user_ns.update(locals)
    app.start()