from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core.util import scaled_integer
def _gsutil_format_datetime_string(datetime_object):
    """Returns datetime in gsutil format, e.g. 'Tue, 08 Jun 2021 21:15:33 GMT'."""
    return datetime_object.strftime('%a, %d %b %Y %H:%M:%S GMT')