from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.spanner.resource_args import CloudKmsKeyName
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.credentials import http
from googlecloudsdk.core.util import times
import six
from six.moves import http_client as httplib
from six.moves import urllib
def ParseExpireTime(expiration_value):
    """Parse flag value into Datetime format for expireTime."""
    datetime = times.ParseDuration(expiration_value).GetRelativeDateTime(times.Now(times.UTC))
    parsed_datetime = times.FormatDateTime(datetime, '%Y-%m-%dT%H:%M:%S.%6f%Ez', tzinfo=times.UTC)
    return parsed_datetime