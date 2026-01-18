import os
import re
import sys
import typing as t
from functools import update_wrapper
from types import ModuleType
from types import TracebackType
from ._compat import _default_text_stderr
from ._compat import _default_text_stdout
from ._compat import _find_binary_writer
from ._compat import auto_wrap_for_ansi
from ._compat import binary_streams
from ._compat import open_stream
from ._compat import should_strip_ansi
from ._compat import strip_ansi
from ._compat import text_streams
from ._compat import WIN
from .globals import resolve_color_default
def _detect_program_name(path: t.Optional[str]=None, _main: t.Optional[ModuleType]=None) -> str:
    """Determine the command used to run the program, for use in help
    text. If a file or entry point was executed, the file name is
    returned. If ``python -m`` was used to execute a module or package,
    ``python -m name`` is returned.

    This doesn't try to be too precise, the goal is to give a concise
    name for help text. Files are only shown as their name without the
    path. ``python`` is only shown for modules, and the full path to
    ``sys.executable`` is not shown.

    :param path: The Python file being executed. Python puts this in
        ``sys.argv[0]``, which is used by default.
    :param _main: The ``__main__`` module. This should only be passed
        during internal testing.

    .. versionadded:: 8.0
        Based on command args detection in the Werkzeug reloader.

    :meta private:
    """
    if _main is None:
        _main = sys.modules['__main__']
    if not path:
        path = sys.argv[0]
    if getattr(_main, '__package__', None) in {None, ''} or (os.name == 'nt' and _main.__package__ == '' and (not os.path.exists(path)) and os.path.exists(f'{path}.exe')):
        return os.path.basename(path)
    py_module = t.cast(str, _main.__package__)
    name = os.path.splitext(os.path.basename(path))[0]
    if name != '__main__':
        py_module = f'{py_module}.{name}'
    return f'python -m {py_module.lstrip('.')}'