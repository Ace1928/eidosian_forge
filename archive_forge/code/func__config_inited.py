import hashlib
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.errors import ConfigError, ExtensionError
import matplotlib as mpl
from matplotlib import _api, mathtext
from matplotlib.rcsetup import validate_float_or_None
def _config_inited(app, config):
    for i, size in enumerate(app.config.mathmpl_srcset):
        if size[-1] == 'x':
            try:
                float(size[:-1])
            except ValueError:
                raise ConfigError(f'Invalid value for mathmpl_srcset parameter: {size!r}. Must be a list of strings with the multiplicative factor followed by an "x".  e.g. ["2.0x", "1.5x"]')
        else:
            raise ConfigError(f'Invalid value for mathmpl_srcset parameter: {size!r}. Must be a list of strings with the multiplicative factor followed by an "x".  e.g. ["2.0x", "1.5x"]')