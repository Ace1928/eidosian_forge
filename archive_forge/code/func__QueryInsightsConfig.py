from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _QueryInsightsConfig(alloydb_messages, insights_config_query_string_length=None, insights_config_query_plans_per_minute=None, insights_config_record_application_tags=None, insights_config_record_client_address=None):
    """Generates the insights config for the instance.

  Args:
    alloydb_messages: module, Message module for the API client.
    insights_config_query_string_length: number, length of the query string to
      be stored.
    insights_config_query_plans_per_minute: number, number of query plans to
      sample every minute.
    insights_config_record_application_tags: boolean, True if application tags
      should be recorded.
    insights_config_record_client_address: boolean, True if client address
      should be recorded.

  Returns:
    alloydb_messages.QueryInsightsInstanceConfig or None
  """
    should_generate_config = any([insights_config_query_string_length is not None, insights_config_query_plans_per_minute is not None, insights_config_record_application_tags is not None, insights_config_record_client_address is not None])
    if not should_generate_config:
        return None
    insights_config = alloydb_messages.QueryInsightsInstanceConfig()
    if insights_config_query_string_length is not None:
        insights_config.queryStringLength = insights_config_query_string_length
    if insights_config_query_plans_per_minute is not None:
        insights_config.queryPlansPerMinute = insights_config_query_plans_per_minute
    if insights_config_record_application_tags is not None:
        insights_config.recordApplicationTags = insights_config_record_application_tags
    if insights_config_record_client_address is not None:
        insights_config.recordClientAddress = insights_config_record_client_address
    return insights_config