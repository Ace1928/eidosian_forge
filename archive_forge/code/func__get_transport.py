from __future__ import absolute_import
from __future__ import print_function
import threading
import httplib2
from six.moves import range  # pylint: disable=redefined-builtin
def _get_transport(self):
    with self._condition:
        while True:
            if self._transports:
                return self._transports.pop()
            self._condition.wait()