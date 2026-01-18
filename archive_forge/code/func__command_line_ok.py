import sys
from distutils.core import Distribution
import warnings
import distutils.core
import distutils.dist
from numpy.distutils.extension import Extension  # noqa: F401
from numpy.distutils.numpy_distribution import NumpyDistribution
from numpy.distutils.command import config, config_compiler, \
from numpy.distutils.misc_util import is_sequence, is_string
def _command_line_ok(_cache=None):
    """ Return True if command line does not contain any
    help or display requests.
    """
    if _cache:
        return _cache[0]
    elif _cache is None:
        _cache = []
    ok = True
    display_opts = ['--' + n for n in Distribution.display_option_names]
    for o in Distribution.display_options:
        if o[1]:
            display_opts.append('-' + o[1])
    for arg in sys.argv:
        if arg.startswith('--help') or arg == '-h' or arg in display_opts:
            ok = False
            break
    _cache.append(ok)
    return ok