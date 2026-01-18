from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def UpdateFallthroughsMap(fallthroughs_map, resource_arg_name, command_level_fallthroughs):
    """Helper to add a single resource's command level fallthroughs."""
    for attribute_name, fallthroughs in six.iteritems(command_level_fallthroughs or {}):
        key = '{}.{}'.format(resource_arg_name, attribute_name)
        fallthroughs_map[key] = fallthroughs