from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def _GetResourceClassStr(messages, resource_class):
    if resource_class is messages.RouterBgp:
        return 'router'
    elif resource_class is messages.RouterBgpPeer:
        return 'peer'
    else:
        raise ValueError('Invalid resource_class value: {0}'.format(resource_class))