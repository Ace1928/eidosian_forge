from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
class NoGrpcInstalled(exceptions.Error):
    """Error that occurs when the grpc module is not installed."""

    def __init__(self):
        super(NoGrpcInstalled, self).__init__('Please ensure that the gRPC module is installed and the environment is correctly configured. Run `sudo pip3 install grpcio` and set the environment variable CLOUDSDK_PYTHON_SITEPACKAGES=1.')