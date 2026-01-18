import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
def _assertEvent(self, observed_event):
    """Raise AssertionError unless observed_event matches the next expected
        event.

        :seealso: expect_request
        :seealso: expect_disconnect
        """
    try:
        expected_event = self._expected_events.pop(0)
    except IndexError:
        raise AssertionError('Mock medium observed event %r, but no more events expected' % (observed_event,))
    if expected_event[0] == 'read response (partial)':
        if observed_event[0] != 'read response':
            raise AssertionError('Mock medium observed event %r, but expected event %r' % (observed_event, expected_event))
    elif observed_event != expected_event:
        raise AssertionError('Mock medium observed event %r, but expected event %r' % (observed_event, expected_event))
    if self._expected_events:
        next_event = self._expected_events[0]
        if next_event[0].startswith('read response'):
            self._mock_request._response = next_event[1]