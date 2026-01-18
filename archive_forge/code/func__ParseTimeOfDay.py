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
def _ParseTimeOfDay(value):
    m = re.match('^(\\d?\\d):00$', value)
    if m:
        hour = int(m.group(1))
        if hour <= 23 and hour >= 0:
            return alloydb_messages.GoogleTypeTimeOfDay(hours=hour)
    raise arg_parsers.ArgumentTypeError('Failed to parse time of day: {0}, expected format: HH:00.'.format(value))