import os
import sys
from importlib import import_module
import click
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
def _invoke_bpython_shell(locals):
    import bpython
    bpython.embed(locals)