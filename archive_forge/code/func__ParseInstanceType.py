from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _ParseInstanceType(alloydb_messages, instance_type):
    if instance_type:
        return alloydb_messages.Instance.InstanceTypeValueValuesEnum.lookup_by_name(instance_type.upper())
    return None