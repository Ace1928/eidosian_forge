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
def _GetDate(alloydb_message):
    """returns google.type.Date date."""

    def Parse(value):
        full_match = re.match('^\\d{4}-\\d{2}-\\d{2}', value)
        if full_match:
            ymd = full_match.group().split('-')
            year = int(ymd[0])
            month = int(ymd[1])
            day = int(ymd[2])
            _ValidateMonthAndDay(month, day, value)
            return alloydb_message.GoogleTypeDate(year=year, month=month, day=day)
        no_year_match = re.match('\\d{2}-\\d{2}', value)
        if no_year_match:
            ymd = no_year_match.group().split('-')
            month = int(ymd[0])
            day = int(ymd[1])
            _ValidateMonthAndDay(month, day, value)
            return alloydb_message.GoogleTypeDate(year=0, month=month, day=day)
        fmt = '"YYYY-MM-DD" or "MM-DD"'
        err_msg = 'Failed to parse date: {}, expected format: {}.'
        raise arg_parsers.ArgumentTypeError(err_msg.format(value, fmt))
    return Parse