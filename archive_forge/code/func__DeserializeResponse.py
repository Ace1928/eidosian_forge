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
def _DeserializeResponse(self, payload):
    """Convert string into Response and content.

        Args:
          payload: Header and body string to be deserialized.

        Returns:
          A Response object
        """
    status_line, payload = payload.split('\n', 1)
    _, status, _ = status_line.split(' ', 2)
    parser = email_parser.Parser()
    msg = parser.parsestr(payload)
    info = dict(msg)
    info['status'] = status
    content = msg.get_payload()
    return http_wrapper.Response(info, content, self.__batch_url)