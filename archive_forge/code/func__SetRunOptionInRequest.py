from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def _SetRunOptionInRequest(run_option, run_schedule, request, messages):
    """Returns request with the run option set."""
    if run_option == 'manual':
        arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1alpha3Crawler.config.adHocRun', messages.GoogleCloudDatacatalogV1alpha3AdhocRun())
    elif run_option == 'scheduled':
        scheduled_run_option = arg_utils.ChoiceToEnum(run_schedule, messages.GoogleCloudDatacatalogV1alpha3ScheduledRun.ScheduledRunOptionValueValuesEnum)
        arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1alpha3Crawler.config.scheduledRun.scheduledRunOption', scheduled_run_option)
    return request