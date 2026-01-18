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
def _method_reply_return(connection, message, method_name, signature, *retval):
    reply = MethodReturnMessage(message)
    try:
        reply.append(*retval, signature=signature)
    except Exception as e:
        logging.basicConfig()
        if signature is None:
            try:
                signature = reply.guess_signature(retval) + ' (guessed)'
            except Exception as e:
                _logger.error('Unable to guess signature for arguments %r: %s: %s', retval, e.__class__, e)
                raise
        _logger.error('Unable to append %r to message with signature %s: %s: %s', retval, signature, e.__class__, e)
        raise
    if not message.get_no_reply():
        connection.send_message(reply)