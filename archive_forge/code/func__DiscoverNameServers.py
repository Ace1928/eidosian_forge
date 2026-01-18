import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def _DiscoverNameServers():
    import sys
    if sys.platform in ('win32', 'nt'):
        from . import win32dns
        defaults['server'] = win32dns.RegistryResolve()
    elif sys.platform == 'darwin':
        ParseOSXSysConfig()
    else:
        return ParseResolvConf()