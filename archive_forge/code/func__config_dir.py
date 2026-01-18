import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def _config_dir():
    """Return per-user configuration directory as unicode string

    By default this is %APPDATA%/breezy on Windows, $XDG_CONFIG_HOME/breezy on
    Mac OS X and Linux. If the breezy config directory doesn't exist but
    the bazaar one (see bazaar_config_dir()) does, use that instead.
    """
    base = os.environ.get('BRZ_HOME')
    if sys.platform == 'win32':
        if base is None:
            base = win32utils.get_appdata_location()
        if base is None:
            raise RuntimeError('Unable to determine AppData location')
    if base is None:
        base = os.environ.get('XDG_CONFIG_HOME')
        if base is None:
            base = osutils.pathjoin(osutils._get_home_dir(), '.config')
    breezy_dir = osutils.pathjoin(base, 'breezy')
    if osutils.isdir(breezy_dir):
        return (breezy_dir, 'breezy')
    bazaar_dir = bazaar_config_dir()
    if osutils.isdir(bazaar_dir):
        trace.mutter('Using Bazaar configuration directory (%s)', bazaar_dir)
        return (bazaar_dir, 'bazaar')
    return (breezy_dir, 'breezy')