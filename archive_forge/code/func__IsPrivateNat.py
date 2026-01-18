from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags
from googlecloudsdk.core import exceptions as core_exceptions
import six
def _IsPrivateNat(nat, compute_holder):
    return nat.type == compute_holder.client.messages.RouterNat.TypeValueValuesEnum.PRIVATE