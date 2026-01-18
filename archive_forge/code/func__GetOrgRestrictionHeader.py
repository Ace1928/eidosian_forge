from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from google.api_core import bidi
from google.rpc import error_details_pb2
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.calliope import base
from googlecloudsdk.core import config
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport as core_transport
from googlecloudsdk.core.credentials import transport
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_proxy_types
import grpc
from six.moves import urllib
import socks
def _GetOrgRestrictionHeader():
    """Returns the org restriction headers to be used."""
    headers = []
    request_org_restriction_headers = properties.VALUES.resource_policy.org_restriction_header.Get()
    if request_org_restriction_headers:
        headers.append(('x-goog-allowed-resources', request_org_restriction_headers))
    return headers