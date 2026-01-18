from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from dateutil import tz
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
def ParseRescheduleType(sql_messages, reschedule_type):
    if reschedule_type:
        return sql_messages.Reschedule.RescheduleTypeValueValuesEnum.lookup_by_name(reschedule_type.upper())
    return None