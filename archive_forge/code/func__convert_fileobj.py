from socket import AF_UNSPEC as _AF_UNSPEC
from ._daemon import (__version__,
def _convert_fileobj(fileobj):
    try:
        return fileobj.fileno()
    except AttributeError:
        return fileobj