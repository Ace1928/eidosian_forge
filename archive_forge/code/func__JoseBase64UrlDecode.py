from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import json
import os
import subprocess
from containerregistry.client import docker_name
def _JoseBase64UrlDecode(message):
    """Perform a JOSE-style base64 decoding of the supplied message.

  This is based on the docker/libtrust version of the similarly named
  function found here:
    https://github.com/docker/libtrust/blob/master/util.go

  Args:
    message: a JOSE-style base64 url-encoded message.
  Raises:
    BadManifestException: a malformed message was supplied.
  Returns:
    The decoded message.
  """
    bytes_msg = message.encode('utf8')
    l = len(bytes_msg)
    if l % 4 == 0:
        pass
    elif l % 4 == 2:
        bytes_msg += b'=='
    elif l % 4 == 3:
        bytes_msg += b'='
    else:
        raise BadManifestException('Malformed JOSE Base64 encoding.')
    return base64.urlsafe_b64decode(bytes_msg).decode('utf8')