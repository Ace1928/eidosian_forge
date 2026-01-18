from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def _GetTimeOfDay(alloydb_message):
    """returns google.type.TimeOfDay time of day."""

    def Parse(value):
        hour_min_sec = value.split(':')
        if len(hour_min_sec) != 3 or not all([item.isdigit() for item in hour_min_sec]):
            raise arg_parsers.ArgumentTypeError('Failed to parse time of day: {0}, expected format HH:MM:SS.\n        '.format(value))
        hour = int(hour_min_sec[0])
        minute = int(hour_min_sec[1])
        second = int(hour_min_sec[2])
        return alloydb_message.GoogleTypeTimeOfDay(hours=hour, minutes=minute, seconds=second)
    return Parse