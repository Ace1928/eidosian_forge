from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def DenyMaintenancePeriod(sql_messages, instance, deny_maintenance_period_start_date=None, deny_maintenance_period_end_date=None, deny_maintenance_period_time='00:00:00'):
    """Generates the deny maintenance period for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    instance: sql_messages.DatabaseInstance, The original instance, if it might
      be needed to generate the deny maintenance period.
    deny_maintenance_period_start_date: date, Date when the deny maintenance
      period begins, i.e., 2020-11-01.
    deny_maintenance_period_end_date: date, Date when the deny maintenance
      period ends, i.e., 2021-01-10.
    deny_maintenance_period_time: Time when the deny maintenance period
      starts/ends, i.e., 05:00:00.

  Returns:
    sql_messages.DenyMaintenancePeriod or None

  Raises:
    argparse.ArgumentError: invalid deny maintenance period specified.
  """
    old_deny_maintenance_period = None
    if instance and instance.settings and instance.settings.denyMaintenancePeriods and instance.settings.denyMaintenancePeriods[0]:
        old_deny_maintenance_period = instance.settings.denyMaintenancePeriods[0]
    deny_maintenance_period = sql_messages.DenyMaintenancePeriod()
    if old_deny_maintenance_period:
        deny_maintenance_period = old_deny_maintenance_period
        if deny_maintenance_period_start_date:
            ValidateDate(deny_maintenance_period_start_date)
            deny_maintenance_period.startDate = deny_maintenance_period_start_date
        if deny_maintenance_period_end_date:
            ValidateDate(deny_maintenance_period_end_date)
            deny_maintenance_period.endDate = deny_maintenance_period_end_date
        if deny_maintenance_period_time:
            ValidTime(deny_maintenance_period_time)
            deny_maintenance_period.time = deny_maintenance_period_time
    else:
        if not (deny_maintenance_period_start_date and deny_maintenance_period_end_date):
            raise argparse.ArgumentError(None, 'There is no deny maintenance period on the instance. To add one, specify values for both start date and end date.')
        ValidateDate(deny_maintenance_period_start_date)
        deny_maintenance_period.startDate = deny_maintenance_period_start_date
        ValidateDate(deny_maintenance_period_end_date)
        deny_maintenance_period.endDate = deny_maintenance_period_end_date
        if deny_maintenance_period_time:
            ValidTime(deny_maintenance_period_time)
            deny_maintenance_period.time = deny_maintenance_period_time
    return deny_maintenance_period