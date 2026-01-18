import os
import re
import sys
def _default_sysroot(cc):
    """ Returns the root of the default SDK for this system, or '/' """
    global _cache_default_sysroot
    if _cache_default_sysroot is not None:
        return _cache_default_sysroot
    contents = _read_output('%s -c -E -v - </dev/null' % (cc,), True)
    in_incdirs = False
    for line in contents.splitlines():
        if line.startswith('#include <...>'):
            in_incdirs = True
        elif line.startswith('End of search list'):
            in_incdirs = False
        elif in_incdirs:
            line = line.strip()
            if line == '/usr/include':
                _cache_default_sysroot = '/'
            elif line.endswith('.sdk/usr/include'):
                _cache_default_sysroot = line[:-12]
    if _cache_default_sysroot is None:
        _cache_default_sysroot = '/'
    return _cache_default_sysroot