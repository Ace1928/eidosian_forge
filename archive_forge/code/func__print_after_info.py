import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
def _print_after_info(line_contents, stream=None):
    if stream is None:
        stream = sys.stdout
    for token in line_contents:
        after_tokens = token.get_after_tokens()
        if after_tokens:
            s = '%s after: %s\n' % (repr(token.tok), '"' + '", "'.join((t.tok for t in token.get_after_tokens())) + '"')
            stream.write(s)
        else:
            stream.write('%s      (NO REQUISITES)' % repr(token.tok))