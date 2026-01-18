from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import io
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import http_encoding
def DecodeFingerprint(fingerprint):
    """Returns the base64 url decoded fingerprint."""
    try:
        decoded_fingerprint = base64.urlsafe_b64decode(http_encoding.Encode(fingerprint))
    except (TypeError, binascii.Error):
        raise calliope_exceptions.InvalidArgumentException('--fingerprint', 'fingerprint cannot be decoded.')
    return decoded_fingerprint