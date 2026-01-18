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
def _message_cb(self, connection, message):
    if not isinstance(message, MethodCallMessage):
        return
    try:
        method_name = message.get_member()
        interface_name = message.get_interface()
        candidate_method, parent_method = _method_lookup(self, method_name, interface_name)
        args = message.get_args_list(**parent_method._dbus_get_args_options)
        keywords = {}
        if parent_method._dbus_out_signature is not None:
            signature = Signature(parent_method._dbus_out_signature)
        else:
            signature = None
        if parent_method._dbus_async_callbacks:
            return_callback, error_callback = parent_method._dbus_async_callbacks
            keywords[return_callback] = lambda *retval: _method_reply_return(connection, message, method_name, signature, *retval)
            keywords[error_callback] = lambda exception: _method_reply_error(connection, message, exception)
        if parent_method._dbus_sender_keyword:
            keywords[parent_method._dbus_sender_keyword] = message.get_sender()
        if parent_method._dbus_path_keyword:
            keywords[parent_method._dbus_path_keyword] = message.get_path()
        if parent_method._dbus_rel_path_keyword:
            path = message.get_path()
            rel_path = path
            for exp in self._locations:
                if exp[0] is connection:
                    if path == exp[1]:
                        rel_path = '/'
                        break
                    if exp[1] == '/':
                        continue
                    if path.startswith(exp[1] + '/'):
                        suffix = path[len(exp[1]):]
                        if len(suffix) < len(rel_path):
                            rel_path = suffix
            rel_path = ObjectPath(rel_path)
            keywords[parent_method._dbus_rel_path_keyword] = rel_path
        if parent_method._dbus_destination_keyword:
            keywords[parent_method._dbus_destination_keyword] = message.get_destination()
        if parent_method._dbus_message_keyword:
            keywords[parent_method._dbus_message_keyword] = message
        if parent_method._dbus_connection_keyword:
            keywords[parent_method._dbus_connection_keyword] = connection
        retval = candidate_method(self, *args, **keywords)
        if parent_method._dbus_async_callbacks:
            return
        if signature is not None:
            signature_tuple = tuple(signature)
            if len(signature_tuple) == 0:
                if retval == None:
                    retval = ()
                else:
                    raise TypeError('%s has an empty output signature but did not return None' % method_name)
            elif len(signature_tuple) == 1:
                retval = (retval,)
            elif isinstance(retval, Sequence):
                pass
            else:
                raise TypeError('%s has multiple output values in signature %s but did not return a sequence' % (method_name, signature))
        elif retval is None:
            retval = ()
        elif isinstance(retval, tuple) and (not isinstance(retval, Struct)):
            pass
        else:
            retval = (retval,)
        _method_reply_return(connection, message, method_name, signature, *retval)
    except Exception as exception:
        _method_reply_error(connection, message, exception)