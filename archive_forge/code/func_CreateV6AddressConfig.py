from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
def CreateV6AddressConfig(self, client):
    return client.messages.AccessConfig(type=client.messages.AccessConfig.TypeValueValuesEnum.DIRECT_IPV6)