from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app import service_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
import six
class IngressSettingError(exceptions.Error):
    """Errors occurring when setting ingress settings."""
    pass