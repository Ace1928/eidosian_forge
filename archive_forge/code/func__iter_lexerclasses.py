import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.lexers._mapping import LEXERS
from pygments.modeline import get_filetype_from_buffer
from pygments.plugin import find_plugin_lexers
from pygments.util import ClassNotFound, itervalues, guess_decode
def _iter_lexerclasses(plugins=True):
    """Return an iterator over all lexer classes."""
    for key in sorted(LEXERS):
        module_name, name = LEXERS[key][:2]
        if name not in _lexer_cache:
            _load_lexers(module_name)
        yield _lexer_cache[name]
    if plugins:
        for lexer in find_plugin_lexers():
            yield lexer