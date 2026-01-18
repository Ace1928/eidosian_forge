import enum
from . import _compiler, _parser
import functools
import copyreg
@functools.lru_cache(_MAXCACHE)
def _compile_repl(repl, pattern):
    return _parser.parse_template(repl, pattern)