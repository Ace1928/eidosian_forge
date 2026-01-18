import json
import urllib3
from sentry_sdk.integrations import Integration
from sentry_sdk.api import set_context
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
@classmethod
def _get_cloud_resource_context(cls):
    cloud_provider = cls.cloud_provider if cls.cloud_provider != '' else CloudResourceContextIntegration._get_cloud_provider()
    if cloud_provider in context_getters.keys():
        return context_getters[cloud_provider]()
    return {}