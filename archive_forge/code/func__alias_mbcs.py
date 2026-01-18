import codecs
import sys
from . import aliases
def _alias_mbcs(encoding):
    try:
        import _winapi
        ansi_code_page = 'cp%s' % _winapi.GetACP()
        if encoding == ansi_code_page:
            import encodings.mbcs
            return encodings.mbcs.getregentry()
    except ImportError:
        pass