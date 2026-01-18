from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import base64
import contextlib
import os
import re
import ssl
import sys
import tempfile
from googlecloudsdk.api_lib.run import gke
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import files
import requests
import six
from six.moves.urllib import parse as urlparse
class RegionalConnectionContext(ConnectionInfo):
    """Context manager to connect a particular Cloud Run region."""

    def __init__(self, region, api_name, version):
        super(RegionalConnectionContext, self).__init__(api_name, version)
        self.region = region

    @property
    def ns_label(self):
        return 'project'

    @property
    def operator(self):
        return 'Cloud Run'

    @property
    def location_label(self):
        return ' region [{{{{bold}}}}{}{{{{reset}}}}]'.format(self.region)

    @contextlib.contextmanager
    def Connect(self):
        global_endpoint = apis.GetEffectiveApiEndpoint(self._api_name, self._version)
        self.endpoint = DeriveRegionalEndpoint(global_endpoint, self.region)
        with _OverrideEndpointOverrides(self._api_name, self.endpoint):
            yield self

    @property
    def supports_one_platform(self):
        return True