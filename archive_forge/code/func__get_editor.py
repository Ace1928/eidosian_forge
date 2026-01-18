import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
def _get_editor():
    """Return sequence of possible editor binaries for the current platform"""
    try:
        yield (os.environ['BRZ_EDITOR'], '$BRZ_EDITOR')
    except KeyError:
        pass
    e = config.GlobalStack().get('editor')
    if e is not None:
        yield (e, bedding.config_path())
    for varname in ('VISUAL', 'EDITOR'):
        if varname in os.environ:
            yield (os.environ[varname], '$' + varname)
    if sys.platform == 'win32':
        for editor in ('wordpad.exe', 'notepad.exe'):
            yield (editor, None)
    else:
        for editor in ['/usr/bin/editor', 'vi', 'pico', 'nano', 'joe']:
            yield (editor, None)