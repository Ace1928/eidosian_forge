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
class EventletBackdoorConfigValueError(Exception):

    def __init__(self, port_range, help_msg, ex):
        msg = _('Invalid backdoor_port configuration %(range)s: %(ex)s. %(help)s') % {'range': port_range, 'ex': ex, 'help': help_msg}
        super(EventletBackdoorConfigValueError, self).__init__(msg)
        self.port_range = port_range