import errno
import math
import select
import socket
import sys
import time
from collections import namedtuple
from ansible.module_utils.six.moves.collections_abc import Mapping
def _syscall_wrapper(func, recalc_timeout, *args, **kwargs):
    """ Wrapper function for syscalls that could fail due to EINTR.
        All functions should be retried if there is time left in the timeout
        in accordance with PEP 475. """
    timeout = kwargs.get('timeout', None)
    if timeout is None:
        expires = None
        recalc_timeout = False
    else:
        timeout = float(timeout)
        if timeout < 0.0:
            expires = None
        else:
            expires = monotonic() + timeout
    args = list(args)
    if recalc_timeout and 'timeout' not in kwargs:
        raise ValueError('Timeout must be in args or kwargs to be recalculated')
    result = _SYSCALL_SENTINEL
    while result is _SYSCALL_SENTINEL:
        try:
            result = func(*args, **kwargs)
        except (OSError, IOError, select.error) as e:
            errcode = None
            if hasattr(e, 'errno'):
                errcode = e.errno
            elif hasattr(e, 'args'):
                errcode = e.args[0]
            is_interrupt = errcode == errno.EINTR or (hasattr(errno, 'WSAEINTR') and errcode == errno.WSAEINTR)
            if is_interrupt:
                if expires is not None:
                    current_time = monotonic()
                    if current_time > expires:
                        raise OSError(errno.ETIMEDOUT)
                    if recalc_timeout:
                        if 'timeout' in kwargs:
                            kwargs['timeout'] = expires - current_time
                continue
            if errcode:
                raise SelectorError(errcode)
            else:
                raise
    return result