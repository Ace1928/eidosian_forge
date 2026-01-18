from __future__ import (absolute_import, division, print_function)
import os
import sys
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ctypes import CDLL, c_char_p, c_int, byref, POINTER, get_errno
def _module_setup():

    def _check_rc(rc):
        if rc < 0:
            errno = get_errno()
            raise OSError(errno, os.strerror(errno))
        return rc
    binary_char_type = type(b'')

    class _to_char_p:

        @classmethod
        def from_param(cls, strvalue):
            if strvalue is not None and (not isinstance(strvalue, binary_char_type)):
                strvalue = to_bytes(strvalue)
            return strvalue
    _funcmap = dict(is_selinux_enabled={}, is_selinux_mls_enabled={}, lgetfilecon_raw=dict(argtypes=[_to_char_p, POINTER(c_char_p)], restype=_check_rc), matchpathcon=dict(argtypes=[_to_char_p, c_int, POINTER(c_char_p)], restype=_check_rc), security_policyvers={}, selinux_getenforcemode=dict(argtypes=[POINTER(c_int)]), security_getenforce={}, lsetfilecon=dict(argtypes=[_to_char_p, _to_char_p], restype=_check_rc), selinux_getpolicytype=dict(argtypes=[POINTER(c_char_p)], restype=_check_rc))
    _thismod = sys.modules[__name__]
    for fname, cfg in _funcmap.items():
        fn = getattr(_selinux_lib, fname, None)
        if not fn:
            raise ImportError('missing selinux function: {0}'.format(fname))
        base_ptr_type = type(POINTER(c_int))
        fn.argtypes = cfg.get('argtypes', None)
        fn.restype = cfg.get('restype', c_int)
        if not fn.argtypes or not any((argtype for argtype in fn.argtypes if type(argtype) is base_ptr_type)):
            setattr(_thismod, fname, fn)
            continue
    unimplemented_funcs = set(_funcmap).difference(dir(_thismod))
    if unimplemented_funcs:
        raise NotImplementedError('implementation is missing functions: {0}'.format(unimplemented_funcs))