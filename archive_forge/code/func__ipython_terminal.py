import os
import sys
from importlib import import_module
import click
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
def _ipython_terminal(locals):
    from IPython.terminal import embed
    embed.TerminalInteractiveShell(user_ns=locals).mainloop()