from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import json
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def ValidateKey(base64_encoded_string, expected_key_length):
    """ValidateKey(s, k) returns None or raises InvalidKeyExceptionNoContext."""
    if expected_key_length < 1:
        raise ValueError('ValidateKey requires expected_key_length > 1.  Got {0}'.format(expected_key_length))
    if len(base64_encoded_string) != expected_key_length:
        raise InvalidKeyExceptionNoContext(base64_encoded_string, 'Key should contain {0} characters (including padding), but is [{1}] characters long.'.format(expected_key_length, len(base64_encoded_string)))
    if base64_encoded_string[-1] != '=':
        raise InvalidKeyExceptionNoContext(base64_encoded_string, "Bad padding.  Keys should end with an '=' character.")
    try:
        base64_encoded_string_as_str = base64_encoded_string.encode('ascii')
    except UnicodeDecodeError:
        raise InvalidKeyExceptionNoContext(base64_encoded_string, 'Key contains non-ascii characters.')
    if not re.match('^[a-zA-Z0-9+/=]*$', base64_encoded_string):
        raise InvalidKeyExceptionNoContext(base64_encoded_string_as_str, "Key contains unexpected characters. Base64 encoded strings contain only letters (upper or lower case), numbers, plusses '+', slashes '/', or equality signs '='.")
    try:
        base64.b64decode(base64_encoded_string_as_str)
    except TypeError as t:
        raise InvalidKeyExceptionNoContext(base64_encoded_string, 'Key is not valid base64: [{0}].'.format(t.message))