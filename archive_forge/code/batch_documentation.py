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
Handles incoming http response to the request in http_request.

            This is intended to be used as a callback function for
            BatchHttpRequest.Add.

            Args:
              http_response: Deserialized http_wrapper.Response object.
              exception: apiclient.errors.HttpError object if an error
                  occurred.

            