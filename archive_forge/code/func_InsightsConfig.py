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
def InsightsConfig(sql_messages, insights_config_query_insights_enabled=None, insights_config_query_string_length=None, insights_config_record_application_tags=None, insights_config_record_client_address=None, insights_config_query_plans_per_minute=None):
    """Generates the insights config for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    insights_config_query_insights_enabled: boolean, True if query insights
      should be enabled.
    insights_config_query_string_length: number, length of the query string to
      be stored.
    insights_config_record_application_tags: boolean, True if application tags
      should be recorded.
    insights_config_record_client_address: boolean, True if client address
      should be recorded.
    insights_config_query_plans_per_minute: number, number of query plans to
      sample every minute.

  Returns:
    sql_messages.InsightsConfig or None
  """
    should_generate_config = any([insights_config_query_insights_enabled is not None, insights_config_query_string_length is not None, insights_config_record_application_tags is not None, insights_config_record_client_address is not None, insights_config_query_plans_per_minute is not None])
    if not should_generate_config:
        return None
    insights_config = sql_messages.InsightsConfig()
    if insights_config_query_insights_enabled is not None:
        insights_config.queryInsightsEnabled = insights_config_query_insights_enabled
    if insights_config_query_string_length is not None:
        insights_config.queryStringLength = insights_config_query_string_length
    if insights_config_record_application_tags is not None:
        insights_config.recordApplicationTags = insights_config_record_application_tags
    if insights_config_record_client_address is not None:
        insights_config.recordClientAddress = insights_config_record_client_address
    if insights_config_query_plans_per_minute is not None:
        insights_config.queryPlansPerMinute = insights_config_query_plans_per_minute
    return insights_config