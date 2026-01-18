import sys
from copy import deepcopy
from datetime import timedelta
from os import environ
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk._compat import datetime_utcnow, duration_in_milliseconds, reraise
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk._types import TYPE_CHECKING
def _get_google_cloud_logs_url(final_time):
    """
    Generates a Google Cloud Logs console URL based on the environment variables
    Arguments:
        final_time {datetime} -- Final time
    Returns:
        str -- Google Cloud Logs Console URL to logs.
    """
    hour_ago = final_time - timedelta(hours=1)
    formatstring = '%Y-%m-%dT%H:%M:%SZ'
    url = 'https://console.cloud.google.com/logs/viewer?project={project}&resource=cloud_function%2Ffunction_name%2F{function_name}%2Fregion%2F{region}&minLogLevel=0&expandAll=false&timestamp={timestamp_end}&customFacets=&limitCustomFacetWidth=true&dateRangeStart={timestamp_start}&dateRangeEnd={timestamp_end}&interval=PT1H&scrollTimestamp={timestamp_end}'.format(project=environ.get('GCP_PROJECT'), function_name=environ.get('FUNCTION_NAME'), region=environ.get('FUNCTION_REGION'), timestamp_end=final_time.strftime(formatstring), timestamp_start=hour_ago.strftime(formatstring))
    return url