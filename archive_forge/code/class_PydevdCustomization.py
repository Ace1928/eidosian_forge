from _pydevd_bundle.pydevd_constants import QUOTED_LINE_PROTOCOL
from _pydev_bundle import pydev_log
import sys
class PydevdCustomization(object):
    DEFAULT_PROTOCOL: str = QUOTED_LINE_PROTOCOL
    DEBUG_MODE: str = ''
    PREIMPORT: str = ''