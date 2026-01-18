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
def _NewId(self):
    """Create a new id.

        Auto incrementing number that avoids conflicts with ids already used.

        Returns:
           A new unique id string.
        """
    return str(next(self.__last_auto_id))