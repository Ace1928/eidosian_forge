import collections
import email.generator as generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import email.parser as email_parser
import itertools
import time
import uuid
import six
from six.moves import http_client
from six.moves import urllib_parse
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
@staticmethod
def _ConvertHeaderToId(header):
    """Convert a Content-ID header value to an id.

        Presumes the Content-ID header conforms to the format that
        _ConvertIdToHeader() returns.

        Args:
          header: A string indicating the Content-ID header value.

        Returns:
          The extracted id value.

        Raises:
          BatchError if the header is not in the expected format.
        """
    if not (header.startswith('<') or header.endswith('>')):
        raise exceptions.BatchError('Invalid value for Content-ID: %s' % header)
    if '+' not in header:
        raise exceptions.BatchError('Invalid value for Content-ID: %s' % header)
    _, request_id = header[1:-1].rsplit('+', 1)
    return urllib_parse.unquote(request_id)