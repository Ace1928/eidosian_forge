from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import subprocess
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
class DNSEndpointOrUseApplicationDefaultCredentialsError(Error):
    """Error for retrieving DNSEndpoint of a cluster that has none."""

    def __init__(self):
        super(DNSEndpointOrUseApplicationDefaultCredentialsError, self).__init__('Only one of --dns-endpoint or USE_APPLICATION_DEFAULT_CREDENTIALS should be set at a time.')