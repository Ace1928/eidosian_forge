from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import random
import re
import socket
import time
import six
from six.moves import urllib
from six.moves import http_client
from boto import config
from boto import UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from gslib.exception import InvalidUrlError
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.constants import XML_PROGRESS_CALLBACKS
from gslib.utils.constants import UTF8
def _QueryServicePos(self, conn, file_length):
    """Queries service to find out what bytes it currently has.

    Args:
      conn: HTTPConnection to use for the query.
      file_length: Total length of the file.

    Returns:
      (service_start, service_end), where the values are inclusive.
      For example, (0, 2) would mean that the service has bytes 0, 1, *and* 2.

    Raises:
      ResumableUploadException if problem querying service.
    """
    resp = self._QueryServiceState(conn, file_length)
    if resp.status == 200:
        return (0, file_length - 1)
    if resp.status != 308:
        raise ResumableUploadException('Got non-308 response (%s) from service state query' % resp.status, ResumableTransferDisposition.START_OVER)
    got_valid_response = False
    range_spec = resp.getheader('range')
    if range_spec:
        m = re.search('bytes=(\\d+)-(\\d+)', range_spec)
        if m:
            service_start = long(m.group(1))
            service_end = long(m.group(2))
            got_valid_response = True
    else:
        return self.SERVICE_HAS_NOTHING
    if not got_valid_response:
        raise ResumableUploadException("Couldn't parse upload service state query response (%s)" % str(resp.getheaders()), ResumableTransferDisposition.START_OVER)
    if conn.debug >= 1:
        self.logger.debug('Service has: Range: %d - %d.', service_start, service_end)
    return (service_start, service_end)