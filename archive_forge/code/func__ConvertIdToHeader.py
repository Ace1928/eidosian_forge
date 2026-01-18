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
def _ConvertIdToHeader(self, request_id):
    """Convert an id to a Content-ID header value.

        Args:
          request_id: String identifier for a individual request.

        Returns:
          A Content-ID header with the id_ encoded into it. A UUID is
          prepended to the value because Content-ID headers are
          supposed to be universally unique.

        """
    return '<%s+%s>' % (self.__base_id, urllib_parse.quote(request_id))