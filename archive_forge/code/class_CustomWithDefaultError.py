from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
class CustomWithDefaultError(RouterError):
    """Raised when custom advertisements are specified with default mode."""

    def __init__(self, messages, resource_class):
        resource_str = _GetResourceClassStr(messages, resource_class)
        error_msg = _CUSTOM_WITH_DEFAULT_ERROR_MESSAGE.format(resource=resource_str)
        super(CustomWithDefaultError, self).__init__(error_msg)