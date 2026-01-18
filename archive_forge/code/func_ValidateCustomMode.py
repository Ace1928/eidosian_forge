from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def ValidateCustomMode(messages, resource_class, resource):
    """Validate that a router/peer is in custom mode."""
    if resource.advertiseMode is not resource_class.AdvertiseModeValueValuesEnum.CUSTOM:
        raise CustomWithDefaultError(messages, resource_class)