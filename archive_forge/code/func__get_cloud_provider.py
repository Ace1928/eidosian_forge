import json
import urllib3
from sentry_sdk.integrations import Integration
from sentry_sdk.api import set_context
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
@classmethod
def _get_cloud_provider(cls):
    if cls._is_aws():
        return CLOUD_PROVIDER.AWS
    if cls._is_gcp():
        return CLOUD_PROVIDER.GCP
    return ''