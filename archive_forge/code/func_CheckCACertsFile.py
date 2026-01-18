from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import struct
import sys
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import http_proxy_types
import httplib2
import six
from six.moves.urllib import parse
import socks
def CheckCACertsFile(ignore_certs):
    """Get and check that CA cert file exists."""
    ca_certs = httplib2.CA_CERTS
    custom_ca_certs = properties.VALUES.core.custom_ca_certs_file.Get()
    if custom_ca_certs:
        ca_certs = custom_ca_certs
    if not os.path.exists(ca_certs):
        error_msg = 'Unable to locate CA certificates file.'
        log.warning(error_msg)
        error_msg += ' [%s]' % ca_certs
        if ignore_certs:
            log.info(error_msg)
        else:
            raise CACertsFileUnavailable(error_msg)
    return ca_certs