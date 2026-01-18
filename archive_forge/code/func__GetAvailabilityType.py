from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def _GetAvailabilityType(self, cp_type, availability_type):
    if availability_type is None:
        return cp_type.AvailabilityTypeValueValuesEnum.SQL_AVAILABILITY_TYPE_UNSPECIFIED
    return cp_type.AvailabilityTypeValueValuesEnum.lookup_by_name(availability_type)