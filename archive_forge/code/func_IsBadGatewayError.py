from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import socket
from apitools.base.py import encoding
from googlecloudsdk.api_lib.runtime_config import exceptions as rtc_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as sdk_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def IsBadGatewayError(error):
    return getattr(error, 'status_code', None) == 502