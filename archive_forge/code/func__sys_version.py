import collections
import os
import re
import sys
import functools
import itertools
def _sys_version(sys_version=None):
    """ Returns a parsed version of Python's sys.version as tuple
        (name, version, branch, revision, buildno, builddate, compiler)
        referring to the Python implementation name, version, branch,
        revision, build number, build date/time as string and the compiler
        identification string.

        Note that unlike the Python sys.version, the returned value
        for the Python version will always include the patchlevel (it
        defaults to '.0').

        The function returns empty strings for tuple entries that
        cannot be determined.

        sys_version may be given to parse an alternative version
        string, e.g. if the version was read from a different Python
        interpreter.

    """
    if sys_version is None:
        sys_version = sys.version
    result = _sys_version_cache.get(sys_version, None)
    if result is not None:
        return result
    if 'IronPython' in sys_version:
        name = 'IronPython'
        if sys_version.startswith('IronPython'):
            match = _ironpython_sys_version_parser.match(sys_version)
        else:
            match = _ironpython26_sys_version_parser.match(sys_version)
        if match is None:
            raise ValueError('failed to parse IronPython sys.version: %s' % repr(sys_version))
        version, alt_version, compiler = match.groups()
        buildno = ''
        builddate = ''
    elif sys.platform.startswith('java'):
        name = 'Jython'
        match = _sys_version_parser.match(sys_version)
        if match is None:
            raise ValueError('failed to parse Jython sys.version: %s' % repr(sys_version))
        version, buildno, builddate, buildtime, _ = match.groups()
        if builddate is None:
            builddate = ''
        compiler = sys.platform
    elif 'PyPy' in sys_version:
        name = 'PyPy'
        match = _pypy_sys_version_parser.match(sys_version)
        if match is None:
            raise ValueError('failed to parse PyPy sys.version: %s' % repr(sys_version))
        version, buildno, builddate, buildtime = match.groups()
        compiler = ''
    else:
        match = _sys_version_parser.match(sys_version)
        if match is None:
            raise ValueError('failed to parse CPython sys.version: %s' % repr(sys_version))
        version, buildno, builddate, buildtime, compiler = match.groups()
        name = 'CPython'
        if builddate is None:
            builddate = ''
        elif buildtime:
            builddate = builddate + ' ' + buildtime
    if hasattr(sys, '_git'):
        _, branch, revision = sys._git
    elif hasattr(sys, '_mercurial'):
        _, branch, revision = sys._mercurial
    else:
        branch = ''
        revision = ''
    l = version.split('.')
    if len(l) == 2:
        l.append('0')
        version = '.'.join(l)
    result = (name, version, branch, revision, buildno, builddate, compiler)
    _sys_version_cache[sys_version] = result
    return result