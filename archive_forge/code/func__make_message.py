from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import version_tuple
from sympy.interactive.printing import init_printing
from sympy.utilities.misc import ARCH
from sympy import *
def _make_message(ipython=True, quiet=False, source=None):
    """Create a banner for an interactive session. """
    from sympy import __version__ as sympy_version
    from sympy import SYMPY_DEBUG
    import sys
    import os
    if quiet:
        return ''
    python_version = '%d.%d.%d' % sys.version_info[:3]
    if ipython:
        shell_name = 'IPython'
    else:
        shell_name = 'Python'
    info = ['ground types: %s' % GROUND_TYPES]
    cache = os.getenv('SYMPY_USE_CACHE')
    if cache is not None and cache.lower() == 'no':
        info.append('cache: off')
    if SYMPY_DEBUG:
        info.append('debugging: on')
    args = (shell_name, sympy_version, python_version, ARCH, ', '.join(info))
    message = '%s console for SymPy %s (Python %s-%s) (%s)\n' % args
    if source is None:
        source = preexec_source
    _source = ''
    for line in source.split('\n')[:-1]:
        if not line:
            _source += '\n'
        else:
            _source += '>>> ' + line + '\n'
    doc_version = sympy_version
    if 'dev' in doc_version:
        doc_version = 'dev'
    else:
        doc_version = '%s/' % doc_version
    message += '\n' + verbose_message % {'source': _source, 'version': doc_version}
    return message