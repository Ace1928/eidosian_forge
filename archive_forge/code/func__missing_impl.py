import sys
import ctypes
from pyglet.util import debug_print
def _missing_impl(interface_name, method_name):
    """Create a callback returning E_NOTIMPL for methods not present on a COMObject."""

    def missing_cb_func(*_):
        assert _debug_com(f'Non-implemented method {method_name} called in {interface_name}')
        return E_NOTIMPL
    return missing_cb_func