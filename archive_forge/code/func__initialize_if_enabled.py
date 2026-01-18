import errno
import gc
import logging
import os
import pprint
import sys
import tempfile
import traceback
import eventlet.backdoor
import greenlet
import yappi
from eventlet.green import socket
from oslo_service._i18n import _
from oslo_service import _options
def _initialize_if_enabled(conf):
    conf.register_opts(_options.eventlet_backdoor_opts)
    backdoor_locals = {'exit': _dont_use_this, 'quit': _dont_use_this, 'fo': _find_objects, 'pgt': _print_greenthreads, 'pnt': _print_nativethreads, 'prof': _capture_profile}
    if conf.backdoor_port is None and conf.backdoor_socket is None:
        return None
    if conf.backdoor_socket is None:
        start_port, end_port = _parse_port_range(str(conf.backdoor_port))
        sock = _listen('localhost', start_port, end_port)
        where_running = sock.getsockname()[1]
    else:
        try:
            backdoor_socket_path = conf.backdoor_socket.format(pid=os.getpid())
        except (KeyError, IndexError, ValueError) as e:
            backdoor_socket_path = conf.backdoor_socket
            LOG.warning('Could not apply format string to eventlet backdoor socket path ({}) - continuing with unformatted path'.format(e))
        sock = _try_open_unix_domain_socket(backdoor_socket_path)
        where_running = backdoor_socket_path

    def displayhook(val):
        if val is not None:
            pprint.pprint(val)
    sys.displayhook = displayhook
    LOG.info('Eventlet backdoor listening on %(where_running)s for process %(pid)d', {'where_running': where_running, 'pid': os.getpid()})
    thread = eventlet.spawn(eventlet.backdoor.backdoor_server, sock, locals=backdoor_locals)
    return (where_running, thread)