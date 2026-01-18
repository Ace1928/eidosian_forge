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
def _ValidateMonthAndDay(month, day, value):
    """Validates value of month and day."""
    if month < 1 or month > 12:
        raise arg_parsers.ArgumentTypeError('Failed to parse date: {0}, invalid month.'.format(value))
    if day < 1 or day > 31:
        raise arg_parsers.ArgumentTypeError('Failed to parse date: {0}, invalid day.'.format(value))
    return