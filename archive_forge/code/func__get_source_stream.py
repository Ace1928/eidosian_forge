from ... import controldir
from ...commands import Command
from ...option import Option, RegistryOption
from . import helpers, load_fastimport
def _get_source_stream(source):
    if source == '-' or source is None:
        import sys
        try:
            stream = sys.stdin.buffer
        except AttributeError:
            stream = helpers.binary_stream(sys.stdin)
    elif source.endswith('.gz'):
        import gzip
        stream = gzip.open(source, 'rb')
    else:
        stream = open(source, 'rb')
    return stream