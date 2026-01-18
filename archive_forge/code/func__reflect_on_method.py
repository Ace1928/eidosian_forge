import sys
import logging
import threading
import traceback
import _dbus_bindings
from dbus import (
from dbus.decorators import method, signal
from dbus.exceptions import (
from dbus.lowlevel import ErrorMessage, MethodReturnMessage, MethodCallMessage
from dbus.proxies import LOCAL_PATH
from dbus._compat import is_py2
def _reflect_on_method(cls, func):
    args = func._dbus_args
    if func._dbus_in_signature:
        in_sig = tuple(Signature(func._dbus_in_signature))
    else:
        in_sig = _VariantSignature()
    if func._dbus_out_signature:
        out_sig = Signature(func._dbus_out_signature)
    else:
        out_sig = []
    reflection_data = '    <method name="%s">\n' % func.__name__
    for pair in zip(in_sig, args):
        reflection_data += '      <arg direction="in"  type="%s" name="%s" />\n' % pair
    for type in out_sig:
        reflection_data += '      <arg direction="out" type="%s" />\n' % type
    reflection_data += '    </method>\n'
    return reflection_data